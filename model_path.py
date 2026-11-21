"""Model path: the SLM turns a question into one grammar-constrained tool call.
Base Qwen2.5-0.5B-Instruct Q4_K_M, NOT fine-tuned yet, few-shot prompt, 2 threads.
Usage: taskset -c 0,1 python3 model_path.py /path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf"""
import sys, time, json, resource, statistics as st
from llama_cpp import Llama, LlamaGrammar
from saarthi import SIGNALS, EVENTS, TOOLS

GRAMMAR = ('root ::= tool " " arg " " win\n'
           'tool ::= ' + ' | '.join(f'"{t}"' for t in TOOLS + ['none']) + '\n'
           'arg ::= ' + ' | '.join(f'"{a}"' for a in list(SIGNALS) + list(EVENTS) + ['-']) + '\n'
           'win ::= "today" | "yesterday" | "last_trip" | "3d" | "7d" | "10d" | "14d" | "30d" | "42d" | "-"\n')

SYSTEM = """You convert a driver's question (English, Hindi or Hinglish) into one vehicle-data tool call.
Output format: tool argument window

Tools:
trend - how a signal changed over time
largest_change - the day a signal changed the most
events - list or count events
before_event - what happened just before an event
compare - city versus highway
anomalies - anything unusual or abnormal
explain - why range or efficiency got worse
none - anything that is not a question about vehicle data, or asks the car to do something

Signals: energy_per_km (range, efficiency, mileage, consumption), batt_temp_c, tyre_bar, hvac_kw (AC), speed_kph, soc_pct (charge), amb_temp_c (outside temperature)
Events: adas_disengage, battery_overheat, harsh_brake
Window: today, yesterday, last_trip, 3d, 7d, 10d, 14d, 30d, 42d, or - if not given"""
SHOTS = [("why is my range so poor since last week", "explain energy_per_km 7d"),
         ("battery temp ka graph dikhao 30 din ka", "trend batt_temp_c 30d"),
         ("how often did ADAS cut out today", "events adas_disengage today"),
         ("what was happening before the overheat alarm", "before_event battery_overheat -"),
         ("does highway driving cost more energy than city", "compare energy_per_km -"),
         ("kuch abnormal tha kya last trip mein", "anomalies - last_trip"),
         ("which day did mileage fall hardest", "largest_change energy_per_km -"),
         ("turn on the headlights", "none - -"),
         ("kitni baar zor se brake laga pichhle 3 din mein", "events harsh_brake 3d"),
         ("mileage kyun gir gaya", "explain energy_per_km -"),
         ("is tyre pressure dropping over two weeks", "trend tyre_bar 14d"),
         ("tell me a story", "none - -")]

def build_prefix():
    p = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
    for q, a in SHOTS: p += f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
    return p

class ModelPath:
    def __init__(self, path, threads=2):
        t = time.perf_counter()
        self.llm = Llama(model_path=path, n_ctx=1024, n_threads=threads, n_threads_batch=threads, verbose=False, use_mlock=False)
        self.grammar = LlamaGrammar.from_string(GRAMMAR, verbose=False); self.prefix = build_prefix()
        self.load_s = time.perf_counter() - t
        t = time.perf_counter(); self.call("warm up"); self.prefix_s = time.perf_counter() - t   # evaluates and caches the prefix
        self.prefix_tokens = len(self.llm.tokenize(self.prefix.encode()))
    def call(self, q):
        t = time.perf_counter()
        o = self.llm(self.prefix + f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n", max_tokens=24, temperature=0,
                     grammar=self.grammar, stop=["<|im_end|>"])
        dt = time.perf_counter() - t; text = o['choices'][0]['text'].strip(); parts = (text.split() + ['-', '-', '-'])[:3]
        return dict(tool=parts[0], arg=parts[1], win=parts[2], s=dt, out_tokens=o['usage']['completion_tokens'],
                    q_tokens=len(self.llm.tokenize(q.encode())) + 8)

if __name__ == '__main__':
    from evaluate import TEST
    from saarthi import Router
    m = ModelPath(sys.argv[1]); router = Router(); rows = []
    for q, tool, arg in TEST:
        r = m.call(q); fast = router.route(q)[0]
        ok_tool = r['tool'] == tool
        ok_full = ok_tool and (tool == 'none' or arg is None or r['arg'] == arg)
        rows.append(dict(q=q, want=tool, want_arg=arg, got=r['tool'], got_arg=r['arg'], ok_tool=ok_tool, ok_full=ok_full,
                         s=r['s'], out_tokens=r['out_tokens'], q_tokens=r['q_tokens'], router_handled=fast is not None))
    ins = [r for r in rows if r['want'] != 'none']; oos = [r for r in rows if r['want'] == 'none']
    deferred = [r for r in ins if not r['router_handled']]
    lat = [r['s'] for r in rows]
    res = dict(model='Qwen2.5-0.5B-Instruct Q4_K_M, base (no fine-tune), 12-shot prompt, grammar-constrained', threads=2,
               load_s=round(m.load_s, 2), prefix_tokens=m.prefix_tokens, prefix_eval_s_once=round(m.prefix_s, 2),
               in_scope=len(ins), tool_correct=sum(r['ok_tool'] for r in ins), full_correct=sum(r['ok_full'] for r in ins),
               deferred_n=len(deferred), deferred_full_correct=sum(r['ok_full'] for r in deferred),
               oos=len(oos), oos_refused=sum(r['got'] == 'none' for r in oos),
               latency_s_median=round(st.median(lat), 3), latency_s_p95=round(sorted(lat)[int(.95 * len(lat))], 3), latency_s_max=round(max(lat), 3),
               out_tokens_mean=round(st.mean(r['out_tokens'] for r in rows), 1), q_tokens_mean=round(st.mean(r['q_tokens'] for r in rows), 1),
               tokens_per_s_overall=round(sum(r['out_tokens'] + r['q_tokens'] for r in rows) / sum(lat), 1),
               peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
    print(json.dumps(res, indent=1)); json.dump(dict(summary=res, rows=rows), open('model_results.json', 'w'), indent=1)
    for r in rows:
        if not r['ok_full']: print(f"MISS want={r['want']}/{r['want_arg']} got={r['got']}/{r['got_arg']} router={r['router_handled']} | {r['q']}")
