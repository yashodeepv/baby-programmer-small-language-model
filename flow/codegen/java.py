"""java.py - IR2 flow -> Java (17+), structured, no dependencies.

    to_java(flow)                  one self-contained source file (the flow class + runtime)
    flow_method(flow)              the flow as a static method, for a project
    RUNTIME_BODY                   the runtime class body (results, calls, arithmetic, JSON)

Java is statically typed, so every expression is typed first, from the same
scope and type rules the IR2 checker uses: `int`, `boolean`, `String`,
`List<Object>` for lists and `Map<String, Object>` for records (the JSON
shape, read through typed accessors). Java's `int` already wraps like the
spec's 32-bit integers and `/`, `%` already truncate toward zero; the
runtime adds the division-by-zero fault, and reductions accumulate in
BigInteger so they stay exact before wrapping.

Only the structured form is generated; a flow that doesn't structure
cleanly (possible only for hand-edited graphs) raises Unstructured.
"""

import json

from flow.check.dataflow import check_dataflow
from flow.check.diagnostics import Report
from flow.check.typing import type_of
from flow.codegen.python import _can_fault
from flow.codegen.structure import Action, CallStmt, Check, IfStmt, Par, Ret, structure
from flow.gen.order import canonical_order
from flow.ir2 import graph as g
from flow.ir2.expr import (All, Any, Arr, ArrLoop, Bin, Bool, Cmp, Const, Fmt, Index, Invalid, Len,
                           Loop, Not, Pluck, Ref, Sel, Str, Un)
from flow.ir2.types import BOOL, INT, STR, ListT, RecordT

RUNTIME_BODY = r'''
    private R() {}

    /** Thrown by a service when a call fails. */
    public static final class ServiceError extends Exception {
        public ServiceError(String message) { super(message); }
    }

    /** The flow fails with this message. */
    public static class Failed extends RuntimeException {
        public Failed(String message) { super(message); }
    }

    /** A service call failed on every attempt. */
    public static final class CallFailed extends Failed {
        public CallFailed(String message) { super(message); }
    }

    /** A runtime fault inside an expression (division by zero, bad index). */
    public static final class Fault extends RuntimeException {
        public Fault(String message) { super(message); }
    }

    public interface Services {
        Map<String, Object> call(String service, String op, Map<String, Object> args, String step) throws ServiceError;
    }

    public interface Store {
        /** op is "create", "update" or "delete". */
        void store(String entity, String op, Map<String, Object> data);
    }

    public interface Notifier {
        /** channel is "email", "slack", "sms" or "webhook". */
        void notify(String channel, String to, String message);
    }

    public interface Flow {
        Map<String, Object> run(Map<String, Object> request, IO io);
    }

    /** What a flow may touch. */
    public static final class IO {
        public final Services services;
        public final Store store;
        public final Notifier notifier;
        public String at;                    // the step running, for fault messages

        public IO(Services services, Store store, Notifier notifier) {
            this.services = services;
            this.store = store;
            this.notifier = notifier;
        }
    }

    public static Map<String, Object> ok(Map<String, Object> response) {
        return result("ok", response, null);
    }

    public static Map<String, Object> failed(String error, Map<String, Object> response) {
        return result("error", response, error);
    }

    private static Map<String, Object> result(String status, Map<String, Object> response, String error) {
        Map<String, Object> r = new LinkedHashMap<>();
        r.put("status", status);
        r.put("response", response);
        r.put("error", error);
        return r;
    }

    /** Run a flow, turning failures into an error result. */
    public static Map<String, Object> execute(Flow flow, Map<String, Object> request, IO io) {
        try {
            return flow.run(request, io);
        } catch (Failed e) {
            return failed(e.getMessage(), null);
        } catch (Fault e) {
            return failed(io.at + ": " + e.getMessage(), null);
        }
    }

    /** Call a service, retrying; throws CallFailed after the last attempt. */
    public static Map<String, Object> call(IO io, String service, String op, Map<String, Object> args,
                                           String step, int retries) {
        ServiceError last = null;
        for (int attempt = 0; attempt <= retries; attempt++) {
            try {
                return io.services.call(service, op, args, step);
            } catch (ServiceError e) {
                last = e;
            }
        }
        throw new CallFailed(step + ": " + service + "." + op + " failed: " + last.getMessage());
    }

    public static Map<String, Object> map(Object... keysAndValues) {
        Map<String, Object> m = new LinkedHashMap<>();
        for (int i = 0; i < keysAndValues.length; i += 2) m.put((String) keysAndValues[i], keysAndValues[i + 1]);
        return m;
    }

    public static List<Object> list(Object... items) { return new ArrayList<>(Arrays.asList(items)); }

    public static int i(Map<String, Object> m, String k) { return ((Number) m.get(k)).intValue(); }
    public static String s(Map<String, Object> m, String k) { return (String) m.get(k); }
    public static boolean b(Map<String, Object> m, String k) { return (Boolean) m.get(k); }
    @SuppressWarnings("unchecked")
    public static List<Object> l(Map<String, Object> m, String k) { return (List<Object>) m.get(k); }
    @SuppressWarnings("unchecked")
    public static Map<String, Object> m(Map<String, Object> m, String k) { return (Map<String, Object>) m.get(k); }
    @SuppressWarnings("unchecked")
    public static Map<String, Object> asMap(Object o) { return (Map<String, Object>) o; }
    @SuppressWarnings("unchecked")
    public static List<Object> asList(Object o) { return (List<Object>) o; }

    public static List<Object> pluck(List<Object> xs, String field) {
        List<Object> out = new ArrayList<>();
        for (Object x : xs) out.add(asMap(x).get(field));
        return out;
    }

    public static boolean eq(Object a, Object b) { return Objects.equals(a, b); }

    public static int div(int a, int b) {
        if (b == 0) throw new Fault("division by zero");
        return a / b;
    }

    public static int mod(int a, int b) {
        if (b == 0) throw new Fault("division by zero");
        return a % b;
    }

    public static Object idx(List<Object> xs, int i) {
        if (i < 0 || i >= xs.size()) throw new Fault("index " + i + " is outside a list of " + xs.size());
        return xs.get(i);
    }

    public static String fmt(Object v) { return String.valueOf(v); }

    public static List<Object> range(int lo, int hi) {
        List<Object> out = new ArrayList<>();
        for (long v = lo; v <= hi; v++) out.add((int) v);
        return out;
    }

    private static boolean keep(String pred, int k, int v) {
        if (pred == null) return true;
        switch (pred) {
            case "even": return v % 2 == 0;
            case "odd": return v % 2 != 0;
            case "gt": return v > k;
            case "lt": return v < k;
            case "divk": return k != 0 && v % k == 0;
            default: throw new IllegalArgumentException(pred);
        }
    }

    /** sum / product / count / min / max over a list, with an optional filter then map. */
    public static int reduce(String op, List<Object> xs, String map, int k, String pred, int pk) {
        if (op.equals("count") && map == null && pred == null) return xs.size();   // any elements, even records
        List<BigInteger> vals = new ArrayList<>();
        for (Object o : xs) {
            int v = ((Number) o).intValue();
            if (!keep(pred, pk, v)) continue;
            BigInteger b = BigInteger.valueOf(v);
            if (map != null) {
                switch (map) {
                    case "square": b = b.multiply(b); break;
                    case "double": b = b.shiftLeft(1); break;
                    case "addk": b = b.add(BigInteger.valueOf(k)); break;
                    case "mulk": b = b.multiply(BigInteger.valueOf(k)); break;
                    case "constk": b = BigInteger.valueOf(k); break;
                    default: throw new IllegalArgumentException(map);
                }
            }
            vals.add(b);
        }
        switch (op) {
            case "count": return vals.size();
            case "sum": {
                BigInteger acc = BigInteger.ZERO;
                for (BigInteger v : vals) acc = acc.add(v);
                return acc.intValue();
            }
            case "product": {
                BigInteger acc = BigInteger.ONE;
                for (BigInteger v : vals) acc = BigInteger.valueOf(acc.multiply(v).intValue());
                return acc.intValue();
            }
            default: {
                if (vals.isEmpty()) return 0;
                BigInteger best = vals.get(0);
                for (BigInteger v : vals) best = op.equals("min") ? best.min(v) : best.max(v);
                return best.intValue();
            }
        }
    }

    // ---- JSON: enough for records, lists, strings, integers, booleans and null ----

    public static Object parseJson(String text) {
        int[] pos = {0};
        Object v = parseValue(text, pos);
        skipWs(text, pos);
        if (pos[0] != text.length()) throw new IllegalArgumentException("trailing characters in JSON");
        return v;
    }

    private static void skipWs(String t, int[] p) {
        while (p[0] < t.length() && Character.isWhitespace(t.charAt(p[0]))) p[0]++;
    }

    private static Object parseValue(String t, int[] p) {
        skipWs(t, p);
        char c = t.charAt(p[0]);
        if (c == '{') {
            Map<String, Object> m = new LinkedHashMap<>();
            p[0]++;
            skipWs(t, p);
            if (t.charAt(p[0]) == '}') { p[0]++; return m; }
            while (true) {
                skipWs(t, p);
                String k = (String) parseValue(t, p);
                skipWs(t, p);
                p[0]++;                                     // ':'
                m.put(k, parseValue(t, p));
                skipWs(t, p);
                if (t.charAt(p[0]++) == '}') return m;
            }
        }
        if (c == '[') {
            List<Object> l = new ArrayList<>();
            p[0]++;
            skipWs(t, p);
            if (t.charAt(p[0]) == ']') { p[0]++; return l; }
            while (true) {
                l.add(parseValue(t, p));
                skipWs(t, p);
                if (t.charAt(p[0]++) == ']') return l;
            }
        }
        if (c == '"') {
            StringBuilder sb = new StringBuilder();
            p[0]++;
            while (true) {
                char ch = t.charAt(p[0]++);
                if (ch == '"') return sb.toString();
                if (ch != '\\') { sb.append(ch); continue; }
                char e = t.charAt(p[0]++);
                switch (e) {
                    case 'n': sb.append('\n'); break;
                    case 't': sb.append('\t'); break;
                    case 'r': sb.append('\r'); break;
                    case 'b': sb.append('\b'); break;
                    case 'f': sb.append('\f'); break;
                    case 'u': sb.append((char) Integer.parseInt(t.substring(p[0], p[0] + 4), 16)); p[0] += 4; break;
                    default: sb.append(e);
                }
            }
        }
        if (t.startsWith("true", p[0])) { p[0] += 4; return Boolean.TRUE; }
        if (t.startsWith("false", p[0])) { p[0] += 5; return Boolean.FALSE; }
        if (t.startsWith("null", p[0])) { p[0] += 4; return null; }
        int start = p[0];
        while (p[0] < t.length() && "+-0123456789.eE".indexOf(t.charAt(p[0])) >= 0) p[0]++;
        String num = t.substring(start, p[0]);
        if (num.contains(".") || num.contains("e") || num.contains("E")) return Double.parseDouble(num);
        long v = Long.parseLong(num);
        return (v >= Integer.MIN_VALUE && v <= Integer.MAX_VALUE) ? (Object) (int) v : (Object) v;
    }

    public static String toJson(Object v) {
        StringBuilder sb = new StringBuilder();
        writeJson(v, sb);
        return sb.toString();
    }

    private static void writeJson(Object v, StringBuilder sb) {
        if (v == null) { sb.append("null"); return; }
        if (v instanceof String) {
            sb.append('"');
            for (char c : ((String) v).toCharArray()) {
                if (c == '"' || c == '\\') sb.append('\\').append(c);
                else if (c < 0x20) sb.append(String.format("\\u%04x", (int) c));
                else sb.append(c);
            }
            sb.append('"');
            return;
        }
        if (v instanceof Map) {
            sb.append('{');
            boolean first = true;
            for (Map.Entry<?, ?> e : ((Map<?, ?>) v).entrySet()) {
                if (!first) sb.append(',');
                first = false;
                writeJson(String.valueOf(e.getKey()), sb);
                sb.append(':');
                writeJson(e.getValue(), sb);
            }
            sb.append('}');
            return;
        }
        if (v instanceof List) {
            sb.append('[');
            boolean first = true;
            for (Object x : (List<?>) v) {
                if (!first) sb.append(',');
                first = false;
                writeJson(x, sb);
            }
            sb.append(']');
            return;
        }
        sb.append(v);
    }

    /** A deep copy through JSON, so a flow can't change its caller's data. */
    public static Object copy(Object v) { return parseJson(toJson(v)); }
'''

RUNTIME_IMPORTS = 'import java.math.BigInteger;\nimport java.util.*;\n'

RESERVED = {
    'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 'const', 'continue',
    'default', 'do', 'double', 'else', 'enum', 'extends', 'final', 'finally', 'float', 'for', 'goto', 'if',
    'implements', 'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new', 'package', 'private',
    'protected', 'public', 'return', 'short', 'static', 'strictfp', 'super', 'switch', 'synchronized', 'this',
    'throw', 'throws', 'transient', 'try', 'void', 'volatile', 'while', 'true', 'false', 'null', 'var', 'record',
    'yield', 'sealed', 'permits', 'request', 'io', 'R',
}
_CMP = {'eq': '==', 'ne': '!=', 'lt': '<', 'le': '<=', 'gt': '>', 'ge': '>='}
_GET = {INT: 'i', STR: 's', BOOL: 'b'}


def var(sid):
    return sid + '_' if sid in RESERVED else sid


def camel(name):
    return ''.join(p[:1].upper() + p[1:] for p in name.replace('-', '_').split('_') if p) or 'Flow'


def jstr(s):
    return json.dumps(s)


def _getter(t):
    if t in _GET:
        return _GET[t]
    return 'l' if isinstance(t, ListT) else 'm'


def _cast(code, t):
    """An Object-typed expression, as t."""
    if t == INT:
        return f'((Number) {code}).intValue()'
    if t == STR:
        return f'((String) {code})'
    if t == BOOL:
        return f'((Boolean) {code})'
    return f'R.asList({code})' if isinstance(t, ListT) else f'R.asMap({code})'


class _Expr:
    def __init__(self, scope):
        self.scope = scope

    def type(self, e):
        return type_of(e, self.scope)

    def __call__(self, e):
        return self.code(e)

    def code(self, e):
        C = self.code
        if isinstance(e, Const):
            return str(e.v)
        if isinstance(e, Str):
            return jstr(e.v)
        if isinstance(e, Bool):
            return 'true' if e.v else 'false'
        if isinstance(e, Ref):
            head = 'request' if e.path[0] == 'input' else var(e.path[0])
            t = self.scope[e.path[0]]
            code = head
            for name in e.path[1:]:
                ft = t.field(name)
                code = f'R.{_getter(ft)}({code}, {jstr(name)})'
                t = ft
            return code
        if isinstance(e, Bin):
            a, b = C(e.a), C(e.b)
            return {'add': f'({a} + {b})', 'sub': f'({a} - {b})', 'mul': f'({a} * {b})',
                    'and': f'({a} & {b})', 'orr': f'({a} | {b})', 'eor': f'({a} ^ {b})',
                    'lsl': f'({a} << ({b} & 31))', 'lsr': f'({a} >>> ({b} & 31))',
                    'div': f'R.div({a}, {b})', 'mod': f'R.mod({a}, {b})'}[e.op]
        if isinstance(e, Un):
            return f'(-{C(e.a)})' if e.op == 'neg' else f'Math.abs({C(e.a)})'
        if isinstance(e, Cmp):
            t = self.type(e.a)
            if e.op in ('eq', 'ne') and t == STR:
                return f'R.eq({C(e.a)}, {C(e.b)})' if e.op == 'eq' else f'!R.eq({C(e.a)}, {C(e.b)})'
            return f'({C(e.a)} {_CMP[e.op]} {C(e.b)})'
        if isinstance(e, Sel):
            return f'({C(e.c)} ? {C(e.a)} : {C(e.b)})'
        if isinstance(e, Not):
            return f'!({C(e.a)})'
        if isinstance(e, All):
            return '(' + ' && '.join(C(x) for x in e.items) + ')'
        if isinstance(e, Any):
            return '(' + ' || '.join(C(x) for x in e.items) + ')'
        if isinstance(e, Len):
            return f'{C(e.a)}.size()'
        if isinstance(e, Pluck):
            return f'R.pluck({C(e.a)}, {jstr(e.field)})'
        if isinstance(e, Fmt):
            return '("" + ' + ' + '.join(jstr(p.v) if isinstance(p, Str) else f'R.fmt({C(p)})'
                                         for p in e.parts) + ')'
        if isinstance(e, Arr):
            return 'R.list(' + ', '.join(C(x) for x in e.items) + ')'
        if isinstance(e, Index):
            return _cast(f'R.idx({C(e.arr)}, {C(e.idx)})', self.type(e))
        if isinstance(e, (ArrLoop, Loop)):
            if isinstance(e, Loop):
                if e.op == 'countdown':
                    return '0'
                src = f'R.range({C(e.lo)}, {C(e.hi)})'
            else:
                src = C(e.arr)
            pred, pk = (e.pred if e.pred else (None, 0))
            return (f'R.reduce({jstr(e.op)}, {src}, {jstr(e.mapf) if e.mapf else "null"}, {e.k}, '
                    f'{jstr(pred) if pred else "null"}, {pk})')
        if isinstance(e, Invalid):
            raise ValueError(f'expression {e.text!r} is not compiled yet ({e.error})')
        raise TypeError(f'java: cannot compile {e!r}')


def _record(d, X):
    if not d:
        return 'R.map()'
    return 'R.map(' + ', '.join(f'{jstr(k)}, {X(v)}' for k, v in d.items()) + ')'


class _Printer:
    def __init__(self, flow):
        self.flow = flow
        self.scopes = check_dataflow(flow, canonical_order(flow), Report())

    def X(self, s, scope=None):
        return _Expr(scope if scope is not None else self.scopes[s.id])

    def ret(self, s):
        X = self.X(s)
        resp = _record(s.response, X) if s.response is not None else 'null'
        if s.status == 'ok':
            return f'return R.ok({resp});'
        return f'return R.failed({jstr(s.id + ": " + (s.title or "ended with error"))}, {resp});'

    def body(self, stmts, pad):
        out = []
        P = lambda line: out.append(pad + line)
        for st in stmts:
            s = st.step
            if getattr(s, 'title', ''):
                P(f'// {s.title}')
            if _can_fault(s):
                P(f'io.at = {jstr(s.id)};')
            if isinstance(st, Action):
                if isinstance(s, g.Compute):
                    scope = dict(self.scopes[s.id])
                    fields = []
                    P(f'{var(s.id)} = new LinkedHashMap<>();')
                    for name, e in s.assign:
                        if fields:
                            scope[s.id] = RecordT(tuple(fields))
                        P(f'{var(s.id)}.put({jstr(name)}, {_Expr(scope)(e)});')
                        fields.append((name, type_of(e, scope)))
                elif isinstance(s, g.Persist):
                    P(f'io.store.store({jstr(s.entity)}, {jstr(s.op)}, {_record(s.data, self.X(s))});')
                else:
                    X = self.X(s)
                    P(f'io.notifier.notify({jstr(s.channel)}, {X(s.to)}, {X(s.message)});')
            elif isinstance(st, Check):
                P(f'if (!({self.X(s)(s.rule)})) {{')
                if st.on_fail is None:
                    P(f'    throw new R.Failed({jstr(s.id + ": " + (s.message or "validation failed"))});')
                else:
                    out += self.body(st.on_fail, pad + '    ')
                P('}')
            elif isinstance(st, CallStmt):
                line = (f'{var(s.id)} = R.call(io, {jstr(s.service)}, {jstr(s.op)}, {_record(s.args, self.X(s))}, '
                        f'{jstr(s.id)}, {s.retries});')
                if st.on_fail is None:
                    P(line)
                else:
                    P('try {')
                    P('    ' + line)
                    P('} catch (R.CallFailed e) {')
                    out += self.body(st.on_fail, pad + '    ')
                    P('}')
            elif isinstance(st, IfStmt):
                then, orelse, neg = st.then, st.orelse, st.negate
                if not then:
                    then, orelse, neg = orelse, [], not neg
                cond = self.X(s)(s.cond)
                P(f'if (!({cond})) {{' if neg else f'if ({cond}) {{')
                out += self.body(then, pad + '    ')
                if orelse:
                    P('} else {')
                    out += self.body(orelse, pad + '    ')
                P('}')
            elif isinstance(st, Par):
                P('// in parallel (run one after another here, in this order)')
                for branch in st.branches:
                    out += self.body(branch, pad)
            elif isinstance(st, Ret):
                P(self.ret(s))
        return out


def _declared(stmts, acc):
    for st in stmts:
        if isinstance(st.step, (g.Compute, g.Call)):
            acc.append(var(st.step.id))
        for sub in ('then', 'orelse', 'on_fail'):
            _declared(getattr(st, sub, None) or [], acc)
        for b in getattr(st, 'branches', []):
            _declared(b, acc)
    return acc


def flow_method(flow, name=None, indent='    '):
    """The flow as `public static Map<String, Object> <name>(Map<String, Object> request, R.IO io)`."""
    stmts = structure(flow)
    name = name or var(flow.name)
    p = _Printer(flow)
    lines = [f'/** {flow.intent or f"The {flow.name} flow."} */',
             f'public static Map<String, Object> {name}(Map<String, Object> request, R.IO io) {{']
    names = list(dict.fromkeys(_declared(stmts, [])))
    if names:
        lines.append('    Map<String, Object> ' + ', '.join(f'{n} = null' for n in names) + ';')
    lines += p.body(stmts, '    ')
    lines.append('}')
    return '\n'.join(indent + x if x else x for x in lines)


def run_method(flow, method):
    return f'''    /** Run the flow once, turning failures into an error result. */
    public static Map<String, Object> run(Map<String, Object> input, R.Services services, R.Store store,
                                          R.Notifier notifier) {{
        R.IO io = new R.IO(services, store, notifier);
        return R.execute({method}::{var(flow.name)}, input, io);
    }}'''


def class_name(flow):
    return camel(flow.name)


def to_java(flow):
    """One source file, <ClassName>.java: the flow class, with the runtime as class R."""
    cls = class_name(flow)
    return (f'// {flow.name} - generated by baby-programmer from IR2. Do not edit by hand.\n'
            f'// Save as {cls}.java; compile with javac {cls}.java\n\n'
            f'{RUNTIME_IMPORTS}\n'
            f'public final class {cls} {{\n    private {cls}() {{}}\n\n'
            f'{flow_method(flow)}\n\n{run_method(flow, cls)}\n}}\n\n'
            f'/** Shared runtime: results, service calls, 32-bit arithmetic, JSON. */\n'
            f'final class R {{{RUNTIME_BODY}}}\n')
