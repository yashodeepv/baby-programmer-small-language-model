"""java_project.py - A system -> a complete Java 17 project (Maven layout, no dependencies).

    <system>/
      pom.xml   README.md   run-tests.sh   run.sh
      src/main/java/<pkg>/
        runtime/R.java        results, service calls, arithmetic, JSON
        Models.java           records for entities, flow inputs, service contracts (+ fromMap/toMap)
        ports/                one typed interface per service, and Services (routes calls to them)
        flows/                one class per flow
        Flows.java            the registry and what triggers each flow
        triggers/             HttpTrigger.java (JDK HttpServer), Events.java, Schedule.java
        adapters/Memory.java  in-memory store, printing notifier, fake services
        App.java              serve | run | event | tick
      src/test/java/<pkg>/    one test class per flow (plain Java, no JUnit), AllTests.java
"""

import json

from flow.codegen.java import RUNTIME_BODY, RUNTIME_IMPORTS, flow_method, var
from flow.codegen.project.common import camel, collect_types, outline, snake, trigger_line
from flow.ir2.types import BOOL, INT, STR, ListT, RecordT

JAVA_RESERVED = {'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 'const',
                 'continue', 'default', 'do', 'double', 'else', 'enum', 'extends', 'final', 'finally', 'float',
                 'for', 'goto', 'if', 'implements', 'import', 'instanceof', 'int', 'interface', 'long', 'native',
                 'new', 'package', 'private', 'protected', 'public', 'return', 'short', 'static', 'strictfp',
                 'super', 'switch', 'synchronized', 'this', 'throw', 'throws', 'transient', 'try', 'void',
                 'volatile', 'while', 'true', 'false', 'null', 'var', 'record', 'yield'}


def ident(name):
    return name + '_' if name in JAVA_RESERVED else name


def jstr(s):
    return json.dumps(s)


def jtype(t, namer, boxed=False):
    if t == INT:
        return 'Integer' if boxed else 'int'
    if t == STR:
        return 'String'
    if t == BOOL:
        return 'Boolean' if boxed else 'boolean'
    if isinstance(t, ListT):
        return f'List<{jtype(t.elem, namer, True)}>'
    return f'Models.{namer.names[t]}'


def from_obj(t, code, namer, depth=0):
    """An Object holding JSON-shaped data -> the typed Java value."""
    if t == INT:
        return f'((Number) {code}).intValue()'
    if t == STR:
        return f'((String) {code})'
    if t == BOOL:
        return f'((Boolean) {code})'
    if isinstance(t, ListT):
        x = f'x{depth}'                         # nested lists need distinct lambda names
        return (f'R.asList({code}).stream().map({x} -> {from_obj(t.elem, x, namer, depth + 1)})'
                f'.collect(Collectors.toList())')
    return f'Models.{namer.names[t]}.fromMap(R.asMap({code}))'


def to_obj(t, code, depth=0):
    """A typed Java value -> JSON-shaped data."""
    if isinstance(t, ListT):
        x = f'x{depth}'
        return f'{code}.stream().map({x} -> (Object) {to_obj(t.elem, x, depth + 1)}).collect(Collectors.toList())'
    if isinstance(t, RecordT):
        return f'{code}.toMap()'
    return code


def default(t, namer):
    if t == INT:
        return '0'
    if t == STR:
        return '""'
    if t == BOOL:
        return 'false'
    if isinstance(t, ListT):
        return 'new ArrayList<>()'
    return f'new Models.{namer.names[t]}(' + ', '.join(default(ft, namer) for _, ft in t.fields) + ')'


def _models(pkg, namer):
    out = [f'package {pkg};', '', 'import java.util.*;', 'import java.util.stream.Collectors;',
           f'import {pkg}.runtime.R;', '',
           '/** Data shapes: entities, flow inputs and service contracts. Flows exchange them as maps',
           ' *  (the JSON shape); fromMap/toMap convert. */', 'public final class Models {', '    private Models() {}']
    for name, rec in namer.defs:
        comps = ', '.join(f'{jtype(t, namer)} {ident(n)}' for n, t in rec.fields)
        out += ['', f'    public record {name}({comps}) {{',
                f'        public static {name} fromMap(Map<String, Object> m) {{',
                f'            return new {name}(' + ', '.join(from_obj(t, f'm.get({jstr(n)})', namer)
                                                        for n, t in rec.fields) + ');',
                '        }', '',
                '        public Map<String, Object> toMap() {',
                '            Map<String, Object> m = new LinkedHashMap<>();']
        out += [f'            m.put({jstr(n)}, {to_obj(t, ident(n))});' for n, t in rec.fields]
        out += ['            return m;', '        }', '    }']
    out.append('}')
    return '\n'.join(out) + '\n'


def _ports(pkg, system, namer):
    files = {}
    for s, ops in sorted(system.services.items()):
        lines = [f'package {pkg}.ports;', '', 'import java.util.*;', f'import {pkg}.Models;',
                 f'import {pkg}.runtime.R;', '', f'/** The {s} service. Throw R.ServiceError when a call fails. */',
                 f'public interface {camel(s)} {{']
        for op, sig in ops.items():
            params = ', '.join(f'{jtype(t, namer)} {ident(n)}' for n, t in sig.input.fields)
            lines.append(f'    {jtype(sig.output, namer)} {ident(op)}({params}) throws R.ServiceError;')
        lines.append('}')
        files[f'ports/{camel(s)}.java'] = '\n'.join(lines) + '\n'
    fields = '\n'.join(f'    public {camel(s)} {ident(s)};' for s in sorted(system.services))
    cases = []
    for s, ops in sorted(system.services.items()):
        for op, sig in ops.items():
            args = ', '.join(from_obj(t, f'args.get({jstr(n)})', namer) for n, t in sig.input.fields)
            cases.append(f'            case {jstr(s + "." + op)}:\n'
                         f'                if ({ident(s)} == null) throw new R.ServiceError("no implementation for service {s}");\n'
                         f'                return {ident(s)}.{ident(op)}({args}).toMap();')
    body = '\n'.join(cases)
    files['ports/Services.java'] = f'''package {pkg}.ports;

import java.util.*;
import java.util.stream.Collectors;
import {pkg}.Models;
import {pkg}.runtime.R;

/** Routes the flows' calls to your service implementations: set the fields, then pass this to the IO. */
public final class Services implements R.Services {{
{fields}

    @Override
    public Map<String, Object> call(String service, String op, Map<String, Object> args, String step)
            throws R.ServiceError {{
        switch (service + "." + op) {{
{body}
            default:
                throw new R.ServiceError("unknown operation " + service + "." + op);
        }}
    }}
}}
'''
    return files


def _memory(pkg, system, namer):
    keys = ', '.join(f'{jstr(e)}, {jstr(rec.fields[0][0])}' for e, rec in system.entities.items())
    fakes = []
    for s, ops in sorted(system.services.items()):
        methods = []
        for op, sig in ops.items():
            params = ', '.join(f'{jtype(t, namer)} {ident(n)}' for n, t in sig.input.fields)
            methods.append(f'        @Override public {jtype(sig.output, namer)} {ident(op)}({params}) {{\n'
                           f'            return {default(sig.output, namer)};\n        }}')
        fakes.append(f'    /** Always succeeds with placeholder values. */\n'
                     f'    public static final class Fake{camel(s)} implements {camel(s)} {{\n'
                     + '\n'.join(methods) + '\n    }')
    wire = '\n'.join(f'        s.{ident(sv)} = new Fake{camel(sv)}();' for sv in sorted(system.services))
    imports = '\n'.join(f'import {pkg}.ports.{camel(sv)};' for sv in sorted(system.services))
    return f'''package {pkg}.adapters;

import java.util.*;
import {pkg}.Models;
import {pkg}.ports.Services;
{imports}
import {pkg}.runtime.R;

/** In-memory adapters: run and try the flows before real systems are connected. */
public final class Memory {{
    private Memory() {{}}

    static final Map<String, String> KEYS = mapOf({keys});

    private static Map<String, String> mapOf(String... kv) {{
        Map<String, String> m = new HashMap<>();
        for (int i = 0; i < kv.length; i += 2) m.put(kv[i], kv[i + 1]);
        return m;
    }}

    /** Keeps records in lists, one list per entity. */
    public static final class MemoryStore implements R.Store {{
        public final Map<String, List<Map<String, Object>>> records = new HashMap<>();

        @Override
        public synchronized void store(String entity, String op, Map<String, Object> data) {{
            List<Map<String, Object>> rows = records.computeIfAbsent(entity, k -> new ArrayList<>());
            String key = KEYS.get(entity);
            if (op.equals("create")) {{ rows.add(new LinkedHashMap<>(data)); return; }}
            rows.removeIf(r -> Objects.equals(r.get(key), data.get(key)));
            if (op.equals("update")) rows.add(new LinkedHashMap<>(data));
        }}
    }}

    /** Prints messages and keeps them in sent. */
    public static final class PrintNotifier implements R.Notifier {{
        public final List<String> sent = new ArrayList<>();

        @Override
        public synchronized void notify(String channel, String to, String message) {{
            sent.add(channel + " " + to + ": " + message);
            System.out.println("[" + channel + " -> " + to + "] " + message);
        }}
    }}

{chr(10).join(fakes)}

    public static Services fakeServices() {{
        Services s = new Services();
{wire}
        return s;
    }}

    public static R.IO io() {{
        return new R.IO(fakeServices(), new MemoryStore(), new PrintNotifier());
    }}
}}
'''


def _flow_class(pkg, flow):
    cls = camel(flow.name)
    return (f'package {pkg}.flows;\n\nimport java.util.*;\nimport {pkg}.runtime.R;\n\n'
            f'/** {flow.name}: {flow.intent or "generated flow"}\n'
            f' *  Trigger: {trigger_line(flow)}   Mode: {flow.mode}\n'
            f' *  Generated by baby-programmer from IR2. Regenerate rather than edit by hand. */\n'
            f'public final class {cls} {{\n    private {cls}() {{}}\n\n{flow_method(flow)}\n}}\n')


def _flows_registry(pkg, system):
    puts = '\n'.join(f'        FLOWS.put({jstr(f.name)}, {camel(f.name)}::{var(f.name)});' for f in system.flows)
    trig = []
    for f in system.flows:
        t = f.trigger()
        cfg = ', '.join(f'{jstr(k)}, {jstr(v)}' for k, v in {'source': t.source, 'mode': f.mode, **t.config}.items())
        trig.append(f'        TRIGGERS.put({jstr(f.name)}, Map.of({cfg}));')
    imports = '\n'.join(f'import {pkg}.flows.{camel(f.name)};' for f in system.flows)
    return f'''package {pkg};

import java.util.*;
{imports}
import {pkg}.runtime.R;

/** Every flow, by name, and what triggers it. */
public final class Flows {{
    private Flows() {{}}

    public static final Map<String, R.Flow> FLOWS = new LinkedHashMap<>();
    public static final Map<String, Map<String, String>> TRIGGERS = new LinkedHashMap<>();

    static {{
{puts}
{chr(10).join(trig)}
    }}

    public static Map<String, Object> run(String name, Map<String, Object> request, R.IO io) {{
        R.Flow flow = FLOWS.get(name);
        if (flow == null) throw new IllegalArgumentException("unknown flow " + name);
        return R.execute(flow, request, io);
    }}
}}
'''


def _triggers(pkg, system):
    builders = []
    for f in system.flows:
        t = f.trigger()
        if t.source != 'schedule':
            continue
        vals = ', '.join(f'{jstr(n)}, ' + ('today' if ft == STR else ('0' if ft == INT else 'false' if ft == BOOL
                                                                      else 'new ArrayList<>()'))
                         for n, ft in t.payload.fields)
        builders.append(f'            case {jstr(f.name)}: return R.map({vals});')
    return {
        'triggers/HttpTrigger.java': f'''package {pkg}.triggers;

import com.sun.net.httpserver.HttpServer;
import java.io.*;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.function.Supplier;
import {pkg}.Flows;
import {pkg}.runtime.R;

/** HTTP entry points for the flows with an HTTP trigger (the JDK's own HttpServer). */
public final class HttpTrigger {{
    private HttpTrigger() {{}}

    public static HttpServer serve(int port, Supplier<R.IO> ioFactory) throws IOException {{
        Map<String, String> routes = new HashMap<>();
        Flows.TRIGGERS.forEach((name, t) -> {{
            if (t.get("source").equals("http")) routes.put(t.get("method") + " " + t.get("path"), name);
        }});
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", port), 0);
        server.createContext("/", ex -> {{
            String name = routes.get(ex.getRequestMethod() + " " + ex.getRequestURI().getPath());
            int code;
            Object body;
            if (name == null) {{
                code = 404;
                body = R.map("error", "no flow here");
            }} else {{
                String text = new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
                Map<String, Object> request;
                try {{
                    request = text.isBlank() ? new LinkedHashMap<>() : R.asMap(R.parseJson(text));
                }} catch (RuntimeException e) {{
                    request = null;
                }}
                if (request == null) {{
                    code = 400;
                    body = R.map("error", "the body must be JSON");
                }} else if (Flows.TRIGGERS.get(name).get("mode").equals("async")) {{
                    Map<String, Object> req = request;
                    new Thread(() -> Flows.run(name, req, ioFactory.get())).start();
                    code = 202;
                    body = R.map("accepted", true);
                }} else {{
                    Map<String, Object> result;
                    try {{
                        result = Flows.run(name, request, ioFactory.get());
                    }} catch (RuntimeException e) {{
                        result = null;
                    }}
                    if (result == null) {{
                        code = 400;
                        body = R.map("error", "missing or malformed field");
                    }} else if (result.get("status").equals("ok")) {{
                        code = 200;
                        body = result.get("response") == null ? R.map() : result.get("response");
                    }} else {{
                        Map<String, Object> err = R.map("error", result.get("error"));
                        if (result.get("response") != null) err.putAll(R.asMap(result.get("response")));
                        code = 422;
                        body = err;
                    }}
                }}
            }}
            byte[] out = R.toJson(body).getBytes(StandardCharsets.UTF_8);
            ex.getResponseHeaders().set("Content-Type", "application/json");
            ex.sendResponseHeaders(code, out.length);
            try (OutputStream os = ex.getResponseBody()) {{ os.write(out); }}
        }});
        server.start();
        routes.forEach((route, name) -> System.out.println("  " + route + "  ->  " + name));
        System.out.println("listening on http://127.0.0.1:" + port + "/");
        return server;
    }}
}}
''',
        'triggers/Events.java': f'''package {pkg}.triggers;

import java.util.*;
import {pkg}.Flows;
import {pkg}.runtime.R;

/** Event entry points: publish(event, payload) runs the flow listening for it. */
public final class Events {{
    private Events() {{}}

    public static Map<String, Object> publish(String event, Map<String, Object> payload, R.IO io) {{
        for (Map.Entry<String, Map<String, String>> e : Flows.TRIGGERS.entrySet()) {{
            Map<String, String> t = e.getValue();
            if (t.get("source").equals("event") && t.get("name").equals(event)) return Flows.run(e.getKey(), payload, io);
        }}
        throw new IllegalArgumentException("no flow listens for event " + event);
    }}
}}
''',
        'triggers/Schedule.java': f'''package {pkg}.triggers;

import java.time.LocalDateTime;
import java.util.*;
import java.util.function.Supplier;
import {pkg}.Flows;
import {pkg}.runtime.R;

/** Scheduled flows: runDue() runs every flow whose cron matches the current minute. */
public final class Schedule {{
    private Schedule() {{}}

    /** The payload a scheduled run receives. Edit to suit. */
    public static Map<String, Object> payloadFor(String name, LocalDateTime when) {{
        String today = when.toLocalDate().toString();
        switch (name) {{
{chr(10).join(builders)}
            default: return R.map();
        }}
    }}

    public static Map<String, Map<String, Object>> runDue(LocalDateTime when, Supplier<R.IO> ioFactory) {{
        Map<String, Map<String, Object>> out = new LinkedHashMap<>();
        Flows.TRIGGERS.forEach((name, t) -> {{
            if (t.get("source").equals("schedule") && cronMatches(t.get("cron"), when))
                out.put(name, Flows.run(name, payloadFor(name, when), ioFactory.get()));
        }});
        return out;
    }}

    /** 5-field cron (minute hour day month weekday); *, *\\/n, a-b, a,b and numbers; weekday 0 is Sunday. */
    public static boolean cronMatches(String cron, LocalDateTime when) {{
        String[] f = cron.trim().split("\\\\s+");
        if (f.length != 5) throw new IllegalArgumentException("expected 5 cron fields: " + cron);
        int[] v = {{when.getMinute(), when.getHour(), when.getDayOfMonth(), when.getMonthValue(),
                   when.getDayOfWeek().getValue() % 7}};
        for (int i = 0; i < 5; i++) if (!fieldMatches(f[i], v[i])) return false;
        return true;
    }}

    private static boolean fieldMatches(String field, int value) {{
        for (String part : field.split(",")) {{
            int step = 1;
            if (part.contains("/")) {{
                String[] ps = part.split("/");
                part = ps[0];
                step = Integer.parseInt(ps[1]);
            }}
            if (part.equals("*")) {{
                if (value % step == 0) return true;
                continue;
            }}
            String[] r = part.split("-");
            int lo = Integer.parseInt(r[0]), hi = r.length > 1 ? Integer.parseInt(r[1]) : lo;
            if (value >= lo && value <= hi && (value - lo) % step == 0) return true;
        }}
        return false;
    }}

    public static void loop(Supplier<R.IO> ioFactory) {{
        Thread t = new Thread(() -> {{
            LocalDateTime last = null;
            while (true) {{
                LocalDateTime now = LocalDateTime.now().withSecond(0).withNano(0);
                if (!now.equals(last)) {{
                    last = now;
                    runDue(now, ioFactory).forEach((n, r) -> System.out.println(now + " " + n + ": " + r.get("status")));
                }}
                try {{ Thread.sleep(5000); }} catch (InterruptedException e) {{ return; }}
            }}
        }});
        t.setDaemon(true);
        t.start();
    }}
}}
''',
    }


def _app(pkg):
    return f'''package {pkg};

import java.time.LocalDateTime;
import java.util.*;
import {pkg}.adapters.Memory;
import {pkg}.runtime.R;
import {pkg}.triggers.*;

/**
 * java -cp build {pkg}.App serve [port]      HTTP triggers, plus the scheduler
 *                             run FLOW 'JSON'    run one flow with a JSON payload
 *                             event NAME 'JSON'  publish an event
 *                             tick               run the scheduled flows due this minute
 * Connect real systems by replacing Memory.io() with your own R.IO.
 */
public final class App {{
    private App() {{}}

    public static void main(String[] args) throws Exception {{
        String command = args.length > 0 ? args[0] : "";
        switch (command) {{
            case "serve":
                Schedule.loop(Memory::io);
                HttpTrigger.serve(args.length > 1 ? Integer.parseInt(args[1]) : 8000, Memory::io);
                break;
            case "run":
                System.out.println(R.toJson(Flows.run(args[1], payload(args), Memory.io())));
                break;
            case "event":
                System.out.println(R.toJson(Events.publish(args[1], payload(args), Memory.io())));
                break;
            case "tick":
                System.out.println(R.toJson(Schedule.runDue(LocalDateTime.now(), Memory::io)));
                break;
            default:
                System.err.println("usage: App serve [port] | run FLOW JSON | event NAME JSON | tick");
                System.exit(1);
        }}
    }}

    private static Map<String, Object> payload(String[] args) {{
        return args.length > 2 ? R.asMap(R.parseJson(args[2])) : new LinkedHashMap<>();
    }}
}}
'''


def _test_class(pkg, flow, scenarios):
    cls = camel(flow.name) + 'Test'
    text = json.dumps(scenarios)
    chunks = [text[i:i + 20000] for i in range(0, len(text), 20000)] or ['{}']
    parts = ',\n        '.join(jstr(c) for c in chunks)
    return f'''package {pkg};

import java.util.*;
import {pkg}.flows.{camel(flow.name)};
import {pkg}.runtime.R;

/** {flow.name}: every generated scenario, replayed. Generated from the reference simulator. */
public final class {cls} {{
    private {cls}() {{}}

    // split so no single string constant nears the class-file limit
    private static final String SCENARIOS = String.join("", new String[] {{
        {parts}
    }});

    /** Returns {{passed, failed}}. */
    @SuppressWarnings("unchecked")
    public static int[] run() {{
        Map<String, Object> scenarios = R.asMap(R.parseJson(SCENARIOS));
        int passed = 0, failed = 0;
        for (Map.Entry<String, Object> entry : scenarios.entrySet()) {{
            Map<String, Object> sc = R.asMap(entry.getValue());
            Map<String, Object> mocks = sc.containsKey("mocks") ? R.asMap(sc.get("mocks")) : new HashMap<>();
            List<Object> effects = new ArrayList<>();
            Map<String, Integer> attempts = new HashMap<>();
            R.Services services = (service, op, args, step) -> {{
                int n = attempts.merge(step, 1, Integer::sum);
                Object m = mocks.get(step);
                List<Object> outcomes = m instanceof List ? (List<Object>) m : List.of(m);
                Map<String, Object> outcome = R.asMap(outcomes.get(Math.min(n, outcomes.size()) - 1));
                effects.add("call " + service + "." + op + " #" + n);
                if (outcome.containsKey("ok")) return R.asMap(R.copy(outcome.get("ok")));
                throw new R.ServiceError((String) outcome.get("error"));
            }};
            R.Store store = (entity, op, data) -> effects.add("persist " + entity + "." + op);
            R.Notifier notifier = (channel, to, message) -> effects.add("notify " + channel + " " + to + ": " + message);
            Map<String, Object> result = R.execute({camel(flow.name)}::{var(flow.name)}, R.asMap(R.copy(sc.get("input"))),
                                                   new R.IO(services, store, notifier));
            Map<String, Object> want = R.asMap(sc.get("expect"));
            boolean ok = Objects.equals(result.get("status"), want.get("status"))
                    && Objects.equals(result.get("response"), want.get("response"))
                    && Objects.equals(effects, want.get("effects"));
            if (ok) {{
                passed++;
                System.out.println("ok   {flow.name} " + entry.getKey());
            }} else {{
                failed++;
                System.out.println("FAIL {flow.name} " + entry.getKey() + ": expected " + R.toJson(want)
                        + ", got " + R.toJson(result) + " with effects " + R.toJson(effects));
            }}
        }}
        return new int[] {{passed, failed}};
    }}

    public static void main(String[] args) {{
        System.exit(run()[1] == 0 ? 0 : 1);
    }}
}}
'''


def _all_tests(pkg, system):
    runs = '\n'.join(f'        add(totals, {camel(f.name)}Test.run());' for f in system.flows)
    return f'''package {pkg};

/** Runs every flow's tests. Exit code 0 when all pass. */
public final class AllTests {{
    private AllTests() {{}}

    private static void add(int[] totals, int[] r) {{
        totals[0] += r[0];
        totals[1] += r[1];
    }}

    public static void main(String[] args) {{
        int[] totals = {{0, 0}};
{runs}
        System.out.println("passed " + totals[0] + " failed " + totals[1]);
        System.exit(totals[1] == 0 && totals[0] > 0 ? 0 : 1);
    }}
}}
'''


def _readme(system, pkg):
    lines = [f'# {system.name}', '']
    if system.description:
        lines += [system.description, '']
    lines += ['Generated by baby-programmer from a Baby Spec. Every flow was verified against the reference',
              'simulator; `src/test` replays those scenarios, so keep them passing when you edit.',
              'Java 17 or newer, no dependencies.', '', '## Flows', '']
    for f in system.flows:
        lines += [f'### {f.name}', '', f'`{trigger_line(f)}`, {f.mode}.', '']
        if f.intent:
            lines += [f.intent, '']
        lines += outline(f) + ['']
    lines += ['## Run', '', '```bash', './run-tests.sh', './run.sh serve',
              f"./run.sh run {system.flows[0].name} '{{...}}'", '```', '',
              'With Maven, `mvn package` builds the jar. The tests are plain Java (no JUnit), so run them',
              f'with `./run-tests.sh` or `java -cp target/classes:target/test-classes {pkg}.AllTests`.', '',
              '## Connect real systems', '',
              f'Implement the interfaces in `src/main/java/{pkg}/ports/`, set them on a `Services`, and build',
              'an `R.IO(services, store, notifier)` in place of `Memory.io()`.', '',
              'Parallel steps run one after another, in the order written. Integers are Java `int`',
              '(money in cents) and wrap on overflow, exactly as in the spec.', '']
    return '\n'.join(lines)


def generate(system):
    pkg = snake(system.name)
    namer = collect_types(system)
    root = system.name
    main = f'{root}/src/main/java/{pkg}'
    test = f'{root}/src/test/java/{pkg}'
    files = {
        f'{root}/pom.xml': f'''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>generated</groupId>
  <artifactId>{system.name}</artifactId>
  <version>0.1.0</version>
  <packaging>jar</packaging>
  <properties>
    <maven.compiler.release>17</maven.compiler.release>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
  </properties>
  <build>
    <plugins>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-jar-plugin</artifactId>
        <version>3.3.0</version>
        <configuration>
          <archive><manifest><mainClass>{pkg}.App</mainClass></manifest></archive>
        </configuration>
      </plugin>
    </plugins>
  </build>
</project>
''',
        f'{root}/README.md': _readme(system, pkg),
        f'{root}/run-tests.sh': f'''#!/bin/sh
# Compile everything and run the generated tests. Needs a JDK (17+).
set -e
cd "$(dirname "$0")"
rm -rf build && mkdir -p build
javac --release 17 -nowarn -d build $(find src -name '*.java')
java -cp build {pkg}.AllTests
''',
        f'{root}/run.sh': f'''#!/bin/sh
# ./run.sh serve [port] | run FLOW 'JSON' | event NAME 'JSON' | tick
set -e
cd "$(dirname "$0")"
mkdir -p build
javac --release 17 -nowarn -d build $(find src/main -name '*.java')
java -cp build {pkg}.App "$@"
''',
        f'{main}/runtime/R.java': (f'package {pkg}.runtime;\n\n{RUNTIME_IMPORTS}\n'
                                   '/** Shared runtime for the generated flows: results, service calls, 32-bit arithmetic, JSON. */\n'
                                   f'public final class R {{{RUNTIME_BODY}}}\n'),
        f'{main}/Models.java': _models(pkg, namer),
        f'{main}/Flows.java': _flows_registry(pkg, system),
        f'{main}/App.java': _app(pkg),
        f'{main}/adapters/Memory.java': _memory(pkg, system, namer),
        f'{test}/AllTests.java': _all_tests(pkg, system),
    }
    for path, content in _ports(pkg, system, namer).items():
        files[f'{main}/{path}'] = content
    for path, content in _triggers(pkg, system).items():
        files[f'{main}/{path}'] = content
    for f in system.flows:
        files[f'{main}/flows/{camel(f.name)}.java'] = _flow_class(pkg, f)
        files[f'{test}/{camel(f.name)}Test.java'] = _test_class(pkg, f, system.scenarios.get(f.name, {}))
    return files
