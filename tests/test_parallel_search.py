"""MCP transport and real Search-o1 CLI/loop tests using scripted model outputs."""
import contextlib
from datetime import datetime, timezone
import importlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import parallel_search


class Response(io.BytesIO):
    def __init__(self, data, content_type='application/json'):
        super().__init__(data)
        self.headers = {'Content-Type': content_type}


class MCPServer:
    def __init__(self, mode=None, text=False, sse=False, shared_url=False):
        self.calls = []
        self.mode, self.text, self.sse = mode, text, sse
        self.shared_url = shared_url

    def __call__(self, request, timeout):
        payload = json.loads(request.data)
        self.calls.append((request, timeout, payload))
        method = payload['method']
        if method == 'notifications/initialized':
            return Response(b'')
        if method == 'initialize':
            result = {'protocolVersion': '2025-03-26'}
        elif method == 'tools/list':
            result = {'tools': [{'name': 'web_search'}, {'name': 'web_fetch'}]}
        else:
            if self.mode == 'timeout':
                raise socket.timeout('fixture timeout')
            if self.mode == 'permission':
                raise HTTPError(request.full_url, 403, 'Forbidden', {}, None)
            if self.mode == 'interrupt':
                raise KeyboardInterrupt()
            name = payload['params']['name']
            args = payload['params']['arguments']
            if name == 'web_search':
                # Distinct pages for each query exercise per-question fetch context.
                suffix = args['search_queries'][0].replace(' ', '-')
                if self.shared_url:
                    suffix = 'shared-page'
                content = {'results': [{'title': 'A source', 'url': 'https://example.org/' + suffix,
                                        'publish_date': None, 'excerpts': ['Search evidence.']}]}
            else:
                text = 'Evidence from fetched page.'
                if self.shared_url:
                    text += ' Relevant to ' + args['search_queries'][0]
                content = {'results': [{'url': url, 'excerpts': [text]}
                                       for url in args['urls']]}
            result = {'content': [{'type': 'text', 'text': json.dumps(content)}]}
            if not self.text:
                result['structuredContent'] = content
            if self.mode == 'quota':
                result = {'isError': True, 'content': [{'type': 'text', 'text': 'Anonymous quota exceeded'}]}
            if self.mode == 'malformed':
                result = {'structuredContent': {'not_results': []}}
        message = {'jsonrpc': '2.0', 'id': payload['id'], 'result': result}
        if method == 'tools/call' and self.mode == 'rpc_error':
            message = {'jsonrpc': '2.0', 'id': payload['id'], 'error': {'code': -32603, 'message': 'fixture'}}
        if method == 'tools/call' and self.mode == 'wrong_id':
            message['id'] = 'wrong'
        raw = json.dumps(message).encode()
        if self.sse:
            raw = b': keepalive\n\nevent: message\ndata: ' + raw + b'\n\n'
        return Response(raw, 'text/event-stream' if self.sse else 'application/json')

    def tool_calls(self):
        return [payload['params'] for _, _, payload in self.calls if payload['method'] == 'tools/call']


class TransportTests(unittest.TestCase):
    def test_resolver(self):
        resolve = parallel_search.resolve_search_provider
        self.assertEqual(resolve('auto', None, None), 'parallel')
        self.assertEqual(resolve('auto', 'saved-cli-key', None), 'bing')
        self.assertEqual(resolve('auto', 'key', 'https://custom.example'), 'bing')
        self.assertEqual(resolve('parallel', 'key', 'https://custom.example'), 'parallel')
        self.assertEqual(resolve('bing', 'key', None), 'bing')
        for choice, endpoint in [('bing', None), ('auto', 'https://custom.example')]:
            with self.assertRaises(ValueError):
                resolve(choice, None, endpoint)

    def test_json_sse_and_text_content(self):
        for sse, text in [(False, False), (True, False), (False, True)]:
            with self.subTest(sse=sse, text=text):
                server = MCPServer(sse=sse, text=text)
                with patch.object(parallel_search, 'urlopen', server):
                    client = parallel_search.ParallelSearch(timeout=7)
                    results = client.search('a factual query', 'conversation-a')
                    url = results['webPages']['value'][0]['url']
                    self.assertIn('Evidence from fetched page.', client.fetch([url], 'a factual query', 'conversation-a')[url])
                self.assertEqual([p['method'] for _, _, p in server.calls][:3],
                                 ['initialize', 'notifications/initialized', 'tools/list'])
                for request, timeout, _ in server.calls:
                    headers = {key.lower(): val for key, val in request.header_items()}
                    self.assertEqual(request.full_url, 'https://search.parallel.ai/mcp')
                    self.assertEqual(headers['user-agent'], parallel_search.USER_AGENT)
                    self.assertEqual(headers['mcp-protocol-version'], '2025-03-26')
                    self.assertNotIn('authorization', headers)
                    self.assertEqual(timeout, 7)

    def test_failures_and_retry_keep_context(self):
        for mode, error in [('timeout', socket.timeout), ('permission', HTTPError),
                            ('rpc_error', RuntimeError), ('quota', RuntimeError),
                            ('malformed', ValueError), ('wrong_id', ValueError),
                            ('interrupt', KeyboardInterrupt)]:
            with self.subTest(mode=mode):
                server = MCPServer(mode=mode)
                with patch.object(parallel_search, 'urlopen', server):
                    client = parallel_search.ParallelSearch()
                    with self.assertRaises(error):
                        client.search('query', 'same-conversation')
                    # Explicit caller retry preserves identity; the client never retries itself.
                    self.assertEqual(len(server.tool_calls()), 1)
                    server.mode = None
                    client.search('query', 'same-conversation')
                self.assertEqual([c['arguments']['session_id'] for c in server.tool_calls()],
                                 ['same-conversation', 'same-conversation'])

    def test_urls_and_fetch_batch_limit(self):
        client = parallel_search.ParallelSearch()
        server = MCPServer()
        with patch.object(parallel_search, 'urlopen', server):
            for url in ['file:///etc/passwd', 'javascript:alert(1)', 'https://user:pass@example.org']:
                with self.assertRaises(ValueError):
                    client.fetch([url], 'query', 'id')
            self.assertEqual(server.calls, [])
            urls = ['https://example.org/' + str(i) for i in range(21)]
            self.assertEqual(set(client.fetch(urls, 'query', 'id')), set(urls))
        self.assertEqual([len(c['arguments']['urls']) for c in server.tool_calls()], [20, 1])


class AgentLoopTests(unittest.TestCase):
    """Run main(), parse_args(), dispatch, and result consumption with a fake LLM.

    Only GPU/model/evaluation modules are replaced; the CLI, search integration,
    Bing response normalizer, context formatting and inference loop run as shipped.
    """
    @classmethod
    def setUpClass(cls):
        cls.doc_prompts, cls.evaluations = [], []

        class Tokenizer:
            pad_token, eos_token, padding_side = None, '<eos>', None

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def apply_chat_template(self, messages, **kwargs):
                return '\n'.join(message['content'] for message in messages)

        class SamplingParams:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class LLM:
            def __init__(self, **kwargs):
                self.turns = {}

            def generate(self, prompts, sampling_params):
                outputs = []
                for prompt in prompts:
                    if hasattr(sampling_params, 'stop'):
                        question = 'A' if 'Question A' in prompt else 'B'
                        turn = self.turns.get(question, 0)
                        self.turns[question] = turn + 1
                        text = (f'<|begin_search_query|>Question {question} fact {turn}<|end_search_query|>'
                                if turn < 2 else 'Final answer: fixture')
                        if getattr(cls, 'live_query', None) and turn == 0:
                            text = '<|begin_search_query|>' + cls.live_query + '<|end_search_query|>'
                    else:
                        cls.doc_prompts.append(prompt)
                        text = 'Final Information: Evidence consumed by reasoning.'
                    outputs.append(types.SimpleNamespace(outputs=[types.SimpleNamespace(text=text)]))
                return outputs

        stubs = {
            'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda: 0)),
            'transformers': types.SimpleNamespace(AutoTokenizer=Tokenizer),
            'vllm': types.SimpleNamespace(LLM=LLM, SamplingParams=SamplingParams),
            'evaluate': types.SimpleNamespace(
                run_evaluation=lambda *args: cls.evaluations.append(args),
                extract_answer=lambda raw, **kwargs: raw),
        }
        cls.modules_patch = patch.dict(sys.modules, stubs)
        cls.modules_patch.start()
        cls.agent = importlib.import_module('run_search_o1')

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop('run_search_o1', None)
        cls.modules_patch.stop()

    def run_agent(self, server, extra=(), questions=('A', 'B'), prepare=None):
        self.doc_prompts.clear()
        self.evaluations.clear()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data/AIME'
            data.mkdir(parents=True)
            (data / 'test.json').write_text(json.dumps([
                {'Question': 'Question ' + q, 'answer': 'fixture'} for q in questions]))
            if prepare:
                prepare(root)
            args = ['run_search_o1.py', '--dataset_name', 'aime', '--split', 'test',
                    '--model_path', 'scripted-fixture', *extra]
            previous = os.getcwd()
            try:
                os.chdir(root)
                with contextlib.ExitStack() as stack:
                    # Keep network routing/certificates, but isolate all provider
                    # credentials and saved home configuration from the caller.
                    isolated = {key: value for key, value in os.environ.items()
                                if key.lower() in ('http_proxy', 'https_proxy', 'all_proxy', 'no_proxy',
                                                   'ssl_cert_file', 'ssl_cert_dir')}
                    isolated['HOME'] = directory
                    stack.enter_context(patch.dict(os.environ, isolated, clear=True))
                    stack.enter_context(patch.object(sys, 'argv', args))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    if server is not None:
                        stack.enter_context(patch.object(parallel_search, 'urlopen', server))
                    self.agent.main()
                caches = {str(p.relative_to(root)): p.read_text() for p in root.glob('cache/**/*.json')}
                return list(self.doc_prompts), list(self.evaluations), caches
            finally:
                os.chdir(previous)

    def test_unconfigured_loop_and_interleaved_conversations(self):
        server = MCPServer()
        docs, evaluations, caches = self.run_agent(server)
        calls = server.tool_calls()
        self.assertEqual([c['name'] for c in calls], ['web_search', 'web_search', 'web_fetch', 'web_fetch'] * 2)
        sessions = {}
        for call in calls:
            args = call['arguments']
            question = args['search_queries'][0].split()[1]
            sessions.setdefault(question, set()).add(args['session_id'])
        self.assertEqual([len(v) for v in sessions.values()], [1, 1])
        self.assertNotEqual(sessions['A'], sessions['B'])
        self.assertEqual(len(docs), 4)
        self.assertTrue(all('Evidence from fetched page.' in doc for doc in docs))
        self.assertTrue(all('Evidence consumed by reasoning.' in text for text in evaluations[0][2]))
        self.assertIn('cache/parallel/search_cache.json', caches)
        # A new invocation creates new conversation IDs even with the same cached client implementation.
        again = MCPServer()
        self.run_agent(again)
        self.assertTrue({c['arguments']['session_id'] for c in calls}.isdisjoint(
            {c['arguments']['session_id'] for c in again.tool_calls()}))
        for request, _, _ in server.calls:
            self.assertEqual(request.full_url, parallel_search.MCP_URL)
            self.assertEqual(request.get_header('User-agent'), parallel_search.USER_AGENT)

    def test_search_disabled_makes_no_mcp_requests(self):
        server = MCPServer()
        self.run_agent(server, extra=['--max_search_limit', '0'])
        self.assertEqual(server.calls, [])

    def test_shared_pages_keep_query_focused_excerpts_separate(self):
        docs, _, caches = self.run_agent(MCPServer(shared_url=True))
        for question, doc in zip(['A', 'B', 'A', 'B'], docs):
            other = 'B' if question == 'A' else 'A'
            self.assertIn('Relevant to Question ' + question, doc)
            self.assertNotIn('Relevant to Question ' + other, doc)
        self.assertEqual(len(json.loads(caches['cache/parallel/url_cache.json'])), 4)

    def test_bing_credentials_override_and_legacy_cache(self):
        def existing_cache(root):
            (root / 'cache').mkdir()
            (root / 'cache/search_cache.json').write_text('{}')
        server = MCPServer()
        with patch.object(self.agent, 'bing_web_search', return_value={}) as bing:
            _, _, caches = self.run_agent(server, extra=['--bing_subscription_key', 'saved-key',
                                                         '--bing_endpoint', 'https://custom.example/search'],
                                          prepare=existing_cache)
            self.assertEqual(bing.call_count, 4)
            self.assertEqual(bing.call_args.args[1:3], ('saved-key', 'https://custom.example/search'))
            self.assertIn('cache/search_cache.json', caches)
        self.assertEqual(server.calls, [])
        with patch.object(self.agent, 'bing_web_search') as bing:
            self.run_agent(MCPServer(), extra=['--search_provider', 'parallel', '--bing_subscription_key', 'key'])
            bing.assert_not_called()

    def test_provider_cache_isolation_and_cached_results(self):
        server = MCPServer()
        def existing_cache(root):
            (root / 'cache').mkdir()
            (root / 'cache/search_cache.json').write_text('{"legacy": "keep"}')
            (root / 'cache/url_cache.json').write_text('{"https://legacy.example": "keep"}')
        _, _, caches = self.run_agent(server, prepare=existing_cache)
        self.assertEqual(caches['cache/search_cache.json'], '{"legacy": "keep"}')
        def parallel_cache(root):
            for name, value in caches.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value)
        cached_server = MCPServer()
        docs, _, _ = self.run_agent(cached_server, prepare=parallel_cache)
        self.assertEqual(cached_server.calls, [])
        self.assertTrue(all('Evidence from fetched page.' in doc for doc in docs))

    def test_loop_errors_are_reported_without_fallback_or_cache_poisoning(self):
        for mode in ['timeout', 'permission', 'rpc_error', 'quota', 'malformed']:
            with self.subTest(mode=mode), patch.object(self.agent, 'bing_web_search') as bing:
                server = MCPServer(mode=mode)
                docs, _, caches = self.run_agent(server, questions=('A',))
                self.assertTrue(all('Search failed:' in doc for doc in docs))
                self.assertEqual(json.loads(caches['cache/parallel/search_cache.json']), {})
                self.assertEqual(len({c['arguments']['session_id'] for c in server.tool_calls()}), 1)
                self.assertEqual(len(server.tool_calls()), 2)
                bing.assert_not_called()

    def test_loop_fetch_failure_keeps_search_evidence(self):
        server = MCPServer()
        original = server.__call__
        def fail_fetch(request, timeout):
            payload = json.loads(request.data)
            if payload.get('params', {}).get('name') == 'web_fetch':
                raise socket.timeout('page fetch timeout')
            return original(request, timeout)
        docs, _, caches = self.run_agent(fail_fetch, questions=('A',))
        self.assertTrue(all('Search evidence.' in doc and 'Page fetch failed:' in doc for doc in docs))
        self.assertEqual(json.loads(caches['cache/parallel/url_cache.json']), {})

    def test_loop_cancellation_propagates(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_agent(MCPServer(mode='interrupt'), questions=('A',))

    @unittest.skipUnless(os.environ.get('SEARCH_O1_LIVE_MCP') == '1', 'opt-in live anonymous MCP caller')
    def test_live_agent_loop(self):
        evidence_path = os.environ.get('SEARCH_O1_LIVE_EVIDENCE')
        requests = []
        original = parallel_search.urlopen

        def record(request, timeout):
            item = {'at': datetime.now(timezone.utc).isoformat(), 'url': request.full_url,
                    'headers': dict(request.header_items()), 'request': json.loads(request.data)}
            requests.append(item)
            try:
                with original(request, timeout=timeout) as response:
                    raw = response.read(parallel_search.MAX_RESPONSE_BYTES + 1)
                    item.update(status=response.status, headers_received=dict(response.headers),
                                response=raw.decode('utf-8'))
                    buffered = Response(raw)
                    buffered.headers = response.headers
                    return buffered
            except HTTPError as error:
                item.update(status=error.code, headers_received=dict(error.headers),
                            response=error.read().decode('utf-8'))
                raise

        type(self).live_query = 'Search-o1 agentic search reasoning paper'
        try:
            with patch.object(parallel_search, 'urlopen', record):
                docs, evaluations, _ = self.run_agent(None, questions=('A',), extra=['--max_search_limit', '1'])
            if evidence_path:
                Path(evidence_path).write_text(json.dumps({'requests': requests, 'documents': docs}, indent=2))
            if any(any(word in doc.lower() for word in ['quota', 'rate limit', '429']) for doc in docs):
                self.skipTest('Live anonymous MCP quota; inspect the retained caller evidence')
            self.assertTrue(any('https://' in doc and 'Search failed:' not in doc for doc in docs), docs)
            self.assertTrue(evaluations)
            self.assertEqual([r['request']['params']['name'] for r in requests
                              if r['request']['method'] == 'tools/call'], ['web_search', 'web_fetch'])
            self.assertTrue(all('Page fetch failed:' not in doc for doc in docs), docs)
        finally:
            type(self).live_query = None


if __name__ == '__main__':
    unittest.main()
