"""Anonymous Parallel Search MCP for Search-o1 (Python 3.9+, stdlib only)."""
import json
import threading
import uuid
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


MCP_URL = 'https://search.parallel.ai/mcp'
USER_AGENT = 'Search-o1/Parallel-MCP (+https://github.com/RUC-NLPIR/Search-o1)'
PROTOCOL_VERSION = '2025-03-26'
MAX_RESPONSE_BYTES = 4 * 1024 * 1024


def resolve_search_provider(provider, bing_key, bing_endpoint):
    """An explicit choice wins; existing Bing credentials/endpoints keep Bing."""
    if provider != 'auto':
        selected = provider
    else:
        selected = 'bing' if bing_key or bing_endpoint else 'parallel'
    if selected == 'bing' and not bing_key:
        raise ValueError('Bing requires --bing_subscription_key.')
    return selected


def _web_url(url):
    parsed = urlsplit(url)
    return parsed.scheme in ('http', 'https') and bool(parsed.hostname) and not parsed.username and not parsed.password


class ParallelSearch:
    """A shared MCP connection; conversation IDs are supplied by each sequence.

    The anonymous endpoint uses stateless Streamable HTTP. Handle JSON and SSE
    responses, but do not open a long-lived GET stream or retry failed searches.
    """

    def __init__(self, timeout=20):
        self.timeout = timeout
        self._ready = False
        self._lock = threading.Lock()
        self._protocol = PROTOCOL_VERSION
        self._session = None  # MCP transport session, never the tool session_id

    def _rpc(self, method, params, notification=False):
        request_id = str(uuid.uuid4())
        payload = {'jsonrpc': '2.0', 'method': method, 'params': params}
        if not notification:
            payload['id'] = request_id
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream',
            'User-Agent': USER_AGENT,
            'MCP-Protocol-Version': self._protocol,
        }
        if self._session:
            headers['Mcp-Session-Id'] = self._session
        request = Request(MCP_URL, json.dumps(payload).encode('utf-8'), headers, method='POST')
        # Context manager closes the response on errors and KeyboardInterrupt.
        with urlopen(request, timeout=self.timeout) as response:
            if method == 'initialize':
                self._session = response.headers.get('Mcp-Session-Id')
            if notification:
                return None
            if 'text/event-stream' in response.headers.get('Content-Type', ''):
                data, total = [], 0
                for line in response:
                    total += len(line)
                    if total > MAX_RESPONSE_BYTES:
                        raise ValueError('Parallel MCP response is too large.')
                    line = line.decode('utf-8').rstrip('\r\n')
                    if not line:
                        if data:
                            message = json.loads('\n'.join(data))
                            if message.get('id') == request_id:
                                break
                            data = []
                    elif line.startswith('data:'):
                        data.append(line[5:].lstrip(' '))
                else:
                    raise ValueError('Parallel MCP stream ended without a response.')
            else:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise ValueError('Parallel MCP response is too large.')
                message = json.loads(raw)
        if message.get('id') != request_id or message.get('jsonrpc') != '2.0':
            raise ValueError('Invalid Parallel MCP response identity.')
        if 'error' in message:
            raise RuntimeError('Parallel MCP error: ' + str(message['error']))
        if 'result' not in message:
            raise ValueError('Parallel MCP response has no result.')
        return message['result']

    def _initialize(self):
        with self._lock:
            if self._ready:
                return
            info = self._rpc('initialize', {
                'protocolVersion': PROTOCOL_VERSION, 'capabilities': {},
                'clientInfo': {'name': 'Search-o1', 'version': '1.0'},
            })
            if info.get('protocolVersion') != PROTOCOL_VERSION:
                raise ValueError('Unsupported Parallel MCP protocol version.')
            self._rpc('notifications/initialized', {}, notification=True)
            tools = self._rpc('tools/list', {})
            names = {tool['name'] for tool in tools.get('tools', [])}
            if not {'web_search', 'web_fetch'} <= names:
                raise ValueError('Parallel MCP search/fetch tools are unavailable.')
            self._ready = True

    def _call(self, name, arguments):
        self._initialize()
        result = self._rpc('tools/call', {'name': name, 'arguments': arguments})
        if result.get('isError'):
            raise RuntimeError('Parallel MCP tool error: ' + json.dumps(result.get('content', [])))
        content = result.get('structuredContent')
        if content is None:
            texts = [item['text'] for item in result.get('content', []) if item.get('type') == 'text']
            content = json.loads('\n'.join(texts))
        if not isinstance(content, dict) or not isinstance(content.get('results'), list):
            raise ValueError('Parallel MCP returned invalid tool results.')
        return content['results']

    def search(self, query, session_id):
        results = self._call('web_search', {
            'objective': query, 'search_queries': [query], 'session_id': session_id,
        })
        # Match the existing search response consumed by extract_relevant_info.
        return {'webPages': {'value': [
            {'name': row.get('title') or '', 'url': row['url'],
             'datePublished': row.get('publish_date') or '',
             'snippet': '\n'.join(row.get('excerpts') or [])}
            for row in results if isinstance(row.get('url'), str) and _web_url(row['url'])
        ]}}

    def fetch(self, urls, query, session_id):
        if any(not _web_url(url) for url in urls):
            raise ValueError('Parallel fetch requires HTTP(S) URLs without credentials.')
        contents = {}
        for start in range(0, len(urls), 20):
            batch = urls[start:start + 20]
            results = self._call('web_fetch', {
                'urls': batch, 'search_queries': [query], 'session_id': session_id,
            })
            for row in results:
                url = row.get('url')
                if url in batch:
                    contents[url] = '\n'.join(row.get('excerpts') or [])
        return contents
