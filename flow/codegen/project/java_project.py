"""java_project.py - A system -> a Spring Boot 3 project (Java 17, Maven).

    <system>/
      pom.xml   README.md
      src/main/java/<pkg>/
        Application.java          @SpringBootApplication, scheduling on
        runtime/R.java            results, service calls, 32-bit arithmetic
        Models.java               records for entities, flow inputs, service contracts
        ports/                    one typed interface per service; ServiceRouter routes flow calls to them
        adapters/                 @Component stand-ins: in-memory store, logging notifier, fake services
        flows/                    one @Service per flow; the logic itself is plain Java
        web/FlowController.java   @RestController: one endpoint per HTTP trigger
        events/                   FlowEvent, and an @EventListener per event trigger
        schedule/ScheduledFlows   @Scheduled per schedule trigger
      src/main/resources/application.properties
      src/test/java/<pkg>/        JUnit 5: every scenario per flow, plus a Spring context + HTTP test
      src/test/resources/scenarios/<flow>.json

Replacing a stand-in is ordinary Spring: declare your own bean of the port's
type and mark it @Primary (or remove the fake).
"""

import json
import re

from flow.codegen.java import RUNTIME_BODY, flow_method
from flow.codegen.project.common import camel, collect_types, outline, snake, trigger_line
from flow.ir2.types import BOOL, INT, STR, ListT, RecordT

SPRING_BOOT = '3.5.0'

JAVA_RESERVED = {'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 'const',
                 'continue', 'default', 'do', 'double', 'else', 'enum', 'extends', 'final', 'finally', 'float',
                 'for', 'goto', 'if', 'implements', 'import', 'instanceof', 'int', 'interface', 'long', 'native',
                 'new', 'package', 'private', 'protected', 'public', 'return', 'short', 'static', 'strictfp',
                 'super', 'switch', 'synchronized', 'this', 'throw', 'throws', 'transient', 'try', 'void',
                 'volatile', 'while', 'true', 'false', 'null', 'var', 'record', 'yield'}

# The runtime without its JSON section: Spring brings Jackson.
RUNTIME = RUNTIME_BODY.split('    // ---- JSON')[0].rstrip() + '\n'


def ident(name):
    return name + '_' if name in JAVA_RESERVED else name


def lower_camel(name):
    c = camel(name)
    return ident(c[:1].lower() + c[1:])


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
    if t == INT:
        return f'((Number) {code}).intValue()'
    if t == STR:
        return f'((String) {code})'
    if t == BOOL:
        return f'((Boolean) {code})'
    if isinstance(t, ListT):
        x = f'x{depth}'
        return (f'R.asList({code}).stream().map({x} -> {from_obj(t.elem, x, namer, depth + 1)})'
                f'.collect(Collectors.toList())')
    return f'Models.{namer.names[t]}.fromMap(R.asMap({code}))'


def to_obj(t, code, depth=0):
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


def spring_cron(cron):
    """Unix 5-field cron -> Spring's 6-field cron (seconds first)."""
    return '0 ' + ' '.join(cron.split())


# --------------------------------------------------------------------------
# main sources
# --------------------------------------------------------------------------

def _models(pkg, namer):
    out = [f'package {pkg};', '', 'import java.util.*;', 'import java.util.stream.Collectors;',
           f'import {pkg}.runtime.R;', '',
           '/** Data shapes: entities, flow inputs and service contracts. Flows exchange them as maps',
           ' *  (the JSON shape); fromMap/toMap convert. */', 'public final class Models {', 'private Models() {}']
    for name, rec in namer.defs:
        comps = ', '.join(f'{jtype(t, namer)} {ident(n)}' for n, t in rec.fields)
        out += [f'public record {name}({comps}) {{',
                f'public static {name} fromMap(Map<String, Object> m) {{',
                f'return new {name}(' + ', '.join(from_obj(t, f'm.get({jstr(n)})', namer) for n, t in rec.fields) + ');',
                '}',
                'public Map<String, Object> toMap() {',
                'Map<String, Object> m = new LinkedHashMap<>();']
        out += [f'm.put({jstr(n)}, {to_obj(t, ident(n))});' for n, t in rec.fields]
        out += ['return m;', '}', '}']
    out.append('}')
    return '\n'.join(out) + '\n'


def _ports(pkg, system, namer):
    files = {}
    for s, ops in sorted(system.services.items()):
        lines = [f'package {pkg}.ports;', '', 'import java.util.*;', f'import {pkg}.Models;',
                 f'import {pkg}.runtime.R;', '',
                 f'/** The {s} service. Throw R.ServiceError when a call fails. Provide your own',
                 ' *  implementation as a Spring bean (mark it @Primary to replace the fake). */',
                 f'public interface {camel(s)} {{']
        for op, sig in ops.items():
            params = ', '.join(f'{jtype(t, namer)} {ident(n)}' for n, t in sig.input.fields)
            lines.append(f'{jtype(sig.output, namer)} {ident(op)}({params}) throws R.ServiceError;')
        lines.append('}')
        files[f'ports/{camel(s)}.java'] = '\n'.join(lines) + '\n'

    svcs = sorted(system.services)
    fields = '\n'.join(f'private final {camel(s)} {ident(s)};' for s in svcs)
    params = ', '.join(f'{camel(s)} {ident(s)}' for s in svcs)
    assigns = '\n'.join(f'this.{ident(s)} = {ident(s)};' for s in svcs)
    cases = []
    for s in svcs:
        for op, sig in system.services[s].items():
            args = ', '.join(from_obj(t, f'args.get({jstr(n)})', namer) for n, t in sig.input.fields)
            cases.append(f'case {jstr(s + "." + op)}: return {ident(s)}.{ident(op)}({args}).toMap();')
    files['ports/ServiceRouter.java'] = f'''package {pkg}.ports;

import java.util.*;
import java.util.stream.Collectors;
import org.springframework.stereotype.Component;
import {pkg}.Models;
import {pkg}.runtime.R;

/** Routes the flows' service calls to the service beans. */
@Component
public class ServiceRouter implements R.Services {{
{fields}

public ServiceRouter({params}) {{
{assigns}
}}

@Override
public Map<String, Object> call(String service, String op, Map<String, Object> args, String step) throws R.ServiceError {{
switch (service + "." + op) {{
{chr(10).join(cases)}
default: throw new R.ServiceError("unknown operation " + service + "." + op);
}}
}}
}}
'''
    return files


def _adapters(pkg, system, namer):
    keys = ', '.join(f'Map.entry({jstr(e)}, {jstr(rec.fields[0][0])})' for e, rec in system.entities.items())
    files = {
        'adapters/InMemoryStore.java': f'''package {pkg}.adapters;

import java.util.*;
import org.springframework.stereotype.Component;
import {pkg}.runtime.R;

/** Keeps records in memory, one list per entity. Replace with a @Primary R.Store bean for a real database. */
@Component
public class InMemoryStore implements R.Store {{
private static final Map<String, String> KEYS = Map.ofEntries({keys});

public final Map<String, List<Map<String, Object>>> records = new HashMap<>();

@Override
public synchronized void store(String entity, String op, Map<String, Object> data) {{
List<Map<String, Object>> rows = records.computeIfAbsent(entity, k -> new ArrayList<>());
String key = KEYS.get(entity);
if (op.equals("create")) {{
rows.add(new LinkedHashMap<>(data));
return;
}}
rows.removeIf(r -> Objects.equals(r.get(key), data.get(key)));
if (op.equals("update")) rows.add(new LinkedHashMap<>(data));
}}
}}
''',
        'adapters/LoggingNotifier.java': f'''package {pkg}.adapters;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import {pkg}.runtime.R;

/** Logs messages instead of sending them. Replace with a @Primary R.Notifier bean. */
@Component
public class LoggingNotifier implements R.Notifier {{
private static final Logger log = LoggerFactory.getLogger(LoggingNotifier.class);

@Override
public void notify(String channel, String to, String message) {{
log.info("[{{}} -> {{}}] {{}}", channel, to, message);
}}
}}
''',
    }
    for s, ops in sorted(system.services.items()):
        methods = []
        for op, sig in ops.items():
            params = ', '.join(f'{jtype(t, namer)} {ident(n)}' for n, t in sig.input.fields)
            methods.append(f'@Override\npublic {jtype(sig.output, namer)} {ident(op)}({params}) {{\n'
                           f'return {default(sig.output, namer)};\n}}')
        files[f'adapters/Fake{camel(s)}.java'] = f'''package {pkg}.adapters;

import java.util.*;
import org.springframework.stereotype.Component;
import {pkg}.Models;
import {pkg}.ports.{camel(s)};

/** Stands in for the {s} service: always succeeds with placeholder values. */
@Component
public class Fake{camel(s)} implements {camel(s)} {{
{chr(10).join(methods)}
}}
'''
    return files


def _flow_service(pkg, flow):
    cls = camel(flow.name) + 'Flow'
    method = lower_camel(flow.name)
    logic = flow_method(flow, name=method, indent='')
    logic = logic.replace('public static Map<String, Object>', 'static Map<String, Object>', 1)
    return f'''package {pkg}.flows;

import java.util.*;
import org.springframework.stereotype.Service;
import {pkg}.ports.ServiceRouter;
import {pkg}.runtime.R;

/**
 * {flow.name}: {flow.intent or "generated flow"}
 * Trigger: {trigger_line(flow)}. Mode: {flow.mode}.
 * Generated by baby-programmer from IR2. Regenerate rather than edit by hand.
 */
@Service
public class {cls} {{
private final R.Services services;
private final R.Store store;
private final R.Notifier notifier;

public {cls}(ServiceRouter services, R.Store store, R.Notifier notifier) {{
this.services = services;
this.store = store;
this.notifier = notifier;
}}

/** Run the flow once; failures come back as an error result. */
public Map<String, Object> run(Map<String, Object> request) {{
return R.execute({cls}::{method}, request, new R.IO(services, store, notifier));
}}

{logic}
}}
'''


def _controller(pkg, system):
    http = [f for f in system.flows if f.trigger().source == 'http']
    if not http:
        return None
    fields, params, assigns, methods = [], [], [], []
    for f in http:
        cls, var = camel(f.name) + 'Flow', lower_camel(f.name)
        fields.append(f'private final {cls} {var};')
        params.append(f'{cls} {var}')
        assigns.append(f'this.{var} = {var};')
        t = f.trigger()
        mapping = {'GET': 'GetMapping', 'POST': 'PostMapping', 'PUT': 'PutMapping', 'DELETE': 'DeleteMapping'}[t.config['method']]
        if f.mode == 'async':
            body = (f'Map<String, Object> request = orEmpty(body);\n'
                    f'executor.execute(() -> {var}.run(request));\n'
                    f'return ResponseEntity.accepted().body(Map.of("accepted", true));')
        else:
            body = f'return respond({var}.run(orEmpty(body)));'
        methods.append(f'/** {f.name}{": " + f.intent if f.intent else ""} */\n'
                       f'@{mapping}({jstr(t.config["path"])})\n'
                       f'public ResponseEntity<Object> {var}(@RequestBody(required = false) Map<String, Object> body) {{\n'
                       f'{body}\n}}')
    imports = '\n'.join(f'import {pkg}.flows.{camel(f.name)}Flow;' for f in http)
    return f'''package {pkg}.web;

import java.util.*;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.core.task.TaskExecutor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
{imports}

/** One endpoint per HTTP-triggered flow. Errors from a flow answer 422 with its response. */
@RestController
public class FlowController {{
private final TaskExecutor executor;
{chr(10).join(fields)}

public FlowController(@Qualifier("applicationTaskExecutor") TaskExecutor executor, {", ".join(params)}) {{
this.executor = executor;
{chr(10).join(assigns)}
}}

{chr(10).join(methods)}

@ExceptionHandler({{ClassCastException.class, NullPointerException.class}})
public ResponseEntity<Object> badRequest(RuntimeException e) {{
return ResponseEntity.badRequest().body(Map.of("error", "missing or malformed field"));
}}

private static Map<String, Object> orEmpty(Map<String, Object> body) {{
return body == null ? new LinkedHashMap<>() : body;
}}

@SuppressWarnings("unchecked")
private static ResponseEntity<Object> respond(Map<String, Object> result) {{
Object response = result.get("response");
if ("ok".equals(result.get("status"))) return ResponseEntity.ok(response == null ? Map.of() : response);
Map<String, Object> error = new LinkedHashMap<>();
error.put("error", result.get("error"));
if (response != null) error.putAll((Map<String, Object>) response);
return ResponseEntity.unprocessableEntity().body(error);
}}
}}
'''


def _events(pkg, system):
    ev = [f for f in system.flows if f.trigger().source == 'event']
    files = {'events/FlowEvent.java': f'''package {pkg}.events;

import java.util.Map;

/** Publish with Spring's ApplicationEventPublisher: publisher.publishEvent(new FlowEvent("name", payload)). */
public record FlowEvent(String name, Map<String, Object> payload) {{}}
'''}
    if not ev:
        return files
    fields, params, assigns, methods = [], [], [], []
    for f in ev:
        cls, var = camel(f.name) + 'Flow', lower_camel(f.name)
        name = f.trigger().config['name']
        fields.append(f'private final {cls} {var};')
        params.append(f'{cls} {var}')
        assigns.append(f'this.{var} = {var};')
        methods.append(f'@EventListener(condition = "#event.name == \'{name}\'")\n'
                       f'public void on{camel(name)}(FlowEvent event) {{\n{var}.run(event.payload());\n}}')
    imports = '\n'.join(f'import {pkg}.flows.{camel(f.name)}Flow;' for f in ev)
    files['events/FlowEventListeners.java'] = f'''package {pkg}.events;

import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;
{imports}

/** Runs the flow listening for each event. */
@Component
public class FlowEventListeners {{
{chr(10).join(fields)}

public FlowEventListeners({", ".join(params)}) {{
{chr(10).join(assigns)}
}}

{chr(10).join(methods)}
}}
'''
    return files


def _schedule(pkg, system):
    sc = [f for f in system.flows if f.trigger().source == 'schedule']
    if not sc:
        return None
    fields, params, assigns, methods = [], [], [], []
    for f in sc:
        cls, var = camel(f.name) + 'Flow', lower_camel(f.name)
        t = f.trigger()
        vals = ', '.join(f'{jstr(n)}, ' + ('LocalDate.now().toString()' if ft == STR else
                                            '0' if ft == INT else 'false' if ft == BOOL else 'new ArrayList<>()')
                         for n, ft in t.payload.fields)
        fields.append(f'private final {cls} {var};')
        params.append(f'{cls} {var}')
        assigns.append(f'this.{var} = {var};')
        methods.append(f'/** Unix cron "{t.config["cron"]}". Edit the payload to suit. */\n'
                       f'@Scheduled(cron = {jstr(spring_cron(t.config["cron"]))})\n'
                       f'public void {var}() {{\n{var}.run(R.map({vals}));\n}}')
    imports = '\n'.join(f'import {pkg}.flows.{camel(f.name)}Flow;' for f in sc)
    return f'''package {pkg}.schedule;

import java.time.LocalDate;
import java.util.*;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import {pkg}.runtime.R;
{imports}

/** Runs each scheduled flow on its cron. */
@Component
public class ScheduledFlows {{
{chr(10).join(fields)}

public ScheduledFlows({", ".join(params)}) {{
{chr(10).join(assigns)}
}}

{chr(10).join(methods)}
}}
'''


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def _flow_test(pkg, flow):
    cls = camel(flow.name) + 'Flow'
    method = lower_camel(flow.name)
    return f'''package {pkg}.flows;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.InputStream;
import java.util.*;
import java.util.stream.Stream;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;
import {pkg}.runtime.R;

/** {flow.name}: every generated scenario, replayed. Generated from the reference simulator. */
class {cls}Test {{
private static final ObjectMapper JSON = new ObjectMapper();

@SuppressWarnings("unchecked")
@TestFactory
Stream<DynamicTest> scenarios() throws Exception {{
Map<String, Object> scenarios;
try (InputStream in = getClass().getResourceAsStream("/scenarios/{flow.name}.json")) {{
scenarios = JSON.readValue(in, Map.class);
}}
return scenarios.entrySet().stream().map(e -> DynamicTest.dynamicTest(e.getKey(), () -> {{
Map<String, Object> sc = (Map<String, Object>) e.getValue();
Map<String, Object> mocks = (Map<String, Object>) sc.getOrDefault("mocks", Map.of());
List<Object> effects = new ArrayList<>();
Map<String, Integer> attempts = new HashMap<>();
R.Services services = (service, op, args, step) -> {{
int n = attempts.merge(step, 1, Integer::sum);
Object m = mocks.get(step);
List<Object> outcomes = m instanceof List ? (List<Object>) m : List.of(m);
Map<String, Object> outcome = (Map<String, Object>) outcomes.get(Math.min(n, outcomes.size()) - 1);
effects.add("call " + service + "." + op + " #" + n);
if (outcome.containsKey("ok")) return JSON.convertValue(outcome.get("ok"), Map.class);
throw new R.ServiceError((String) outcome.get("error"));
}};
R.Store store = (entity, op, data) -> effects.add("persist " + entity + "." + op);
R.Notifier notifier = (channel, to, message) -> effects.add("notify " + channel + " " + to + ": " + message);
Map<String, Object> input = JSON.convertValue(sc.get("input"), Map.class);
Map<String, Object> result = R.execute({cls}::{method}, input, new R.IO(services, store, notifier));
Map<String, Object> want = (Map<String, Object>) sc.get("expect");
assertEquals(want.get("status"), result.get("status"), "status");
assertEquals(want.get("response"), result.get("response"), "response");
assertEquals(want.get("effects"), effects, "effects");
}}));
}}
}}
'''


def _app_test(pkg, system):
    http = [f for f in system.flows if f.trigger().source == 'http']
    checks = []
    for f in http:
        t = f.trigger()
        scenarios = system.scenarios.get(f.name, {})
        if not scenarios:
            continue
        payload = json.dumps(next(iter(scenarios.values()))['input'])
        verb = {'GET': 'get', 'POST': 'post', 'PUT': 'put', 'DELETE': 'delete'}[t.config['method']]
        matcher = 'status().isAccepted()' if f.mode == 'async' else 'status().is(anyOf(equalTo(200), equalTo(422)))'
        checks.append(f'mvc.perform({verb}({jstr(t.config["path"])}).contentType(MediaType.APPLICATION_JSON)'
                      f'.content({jstr(payload)})).andExpect({matcher});')
    body = '\n'.join(checks) or '// no HTTP-triggered flows'
    return f'''package {pkg};

import static org.hamcrest.Matchers.anyOf;
import static org.hamcrest.Matchers.equalTo;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

/** The application starts, and every HTTP trigger answers (with the fake services in place). */
@SpringBootTest
@AutoConfigureMockMvc
class ApplicationTests {{
@Autowired
private MockMvc mvc;

@Test
void httpTriggersAnswer() throws Exception {{
{body}
}}
}}
'''


def _readme(system, pkg):
    lines = [f'# {system.name}', '']
    if system.description:
        lines += [system.description, '']
    lines += ['A Spring Boot 3 application generated by baby-programmer from a Baby Spec. Every flow was',
              'verified against the reference simulator; `src/test` replays those scenarios, so keep them',
              'passing when you edit. Java 17+.', '', '## Flows', '']
    for f in system.flows:
        lines += [f'### {f.name}', '', f'`{trigger_line(f)}`, {f.mode}.', '']
        if f.intent:
            lines += [f.intent, '']
        lines += outline(f) + ['']
    lines += ['## Run', '', '```bash', 'mvn test', 'mvn spring-boot:run            # http://localhost:8080', '```', '',
              '## Connect real systems', '',
              'Each outside system is a Spring bean with a stand-in:', '',
              f'- services: implement the interfaces in `src/main/java/{pkg}/ports/` (the `Fake*` classes in',
              '  `adapters/` stand in until you do);',
              '- storage: provide an `R.Store` bean (the stand-in is `InMemoryStore`);',
              '- notifications: provide an `R.Notifier` bean (the stand-in is `LoggingNotifier`).', '',
              'Mark your bean `@Primary`, or delete the stand-in.', '',
              'Parallel steps run one after another, in the order written. Integers are Java `int`',
              '(money in cents) and wrap on overflow, exactly as in the spec. Scheduled flows use the',
              'spec\'s cron with a leading seconds field added, as Spring requires.', '']
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
  <parent>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-parent</artifactId>
    <version>{SPRING_BOOT}</version>
    <relativePath/>
  </parent>
  <groupId>generated</groupId>
  <artifactId>{system.name}</artifactId>
  <version>0.1.0</version>
  <description>{(system.description or 'Generated by baby-programmer')[:200]}</description>
  <properties>
    <java.version>17</java.version>
  </properties>
  <dependencies>
    <dependency>
      <groupId>org.springframework.boot</groupId>
      <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    <dependency>
      <groupId>org.springframework.boot</groupId>
      <artifactId>spring-boot-starter-test</artifactId>
      <scope>test</scope>
    </dependency>
  </dependencies>
  <build>
    <plugins>
      <plugin>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-maven-plugin</artifactId>
      </plugin>
    </plugins>
  </build>
</project>
''',
        f'{root}/README.md': _readme(system, pkg),
        f'{root}/src/main/resources/application.properties': f'spring.application.name={system.name}\nspring.main.banner-mode=off\n',
        f'{main}/Application.java': f'''package {pkg};

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

/** {system.name} - generated by baby-programmer. */
@SpringBootApplication
@EnableScheduling
public class Application {{
public static void main(String[] args) {{
SpringApplication.run(Application.class, args);
}}
}}
''',
        f'{main}/runtime/R.java': (f'package {pkg}.runtime;\n\nimport java.math.BigInteger;\nimport java.util.*;\n\n'
                                   '/** Shared runtime for the generated flows: results, service calls, 32-bit arithmetic. */\n'
                                   f'public final class R {{{RUNTIME}}}\n'),
        f'{main}/Models.java': _models(pkg, namer),
        f'{test}/ApplicationTests.java': _app_test(pkg, system),
    }
    for path, content in {**_ports(pkg, system, namer), **_adapters(pkg, system, namer), **_events(pkg, system)}.items():
        files[f'{main}/{path}'] = content
    controller = _controller(pkg, system)
    if controller:
        files[f'{main}/web/FlowController.java'] = controller
    schedule = _schedule(pkg, system)
    if schedule:
        files[f'{main}/schedule/ScheduledFlows.java'] = schedule
    for f in system.flows:
        files[f'{main}/flows/{camel(f.name)}Flow.java'] = _flow_service(pkg, f)
        files[f'{test}/flows/{camel(f.name)}FlowTest.java'] = _flow_test(pkg, f)
        files[f'{root}/src/test/resources/scenarios/{f.name}.json'] = json.dumps(
            system.scenarios.get(f.name, {}), indent=2) + '\n'
    return files
