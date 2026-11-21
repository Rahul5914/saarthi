"""Generate six weeks of synthetic vehicle history with injected faults and
build the three storage tiers (raw Parquet, 1 s aggregate, trips + events)."""
import duckdb, numpy as np, pandas as pd, os, shutil
rng = np.random.default_rng(7)
OUT, DB, DAYS, HZ = 'store', 'veh.duckdb', 42, 10
START = pd.Timestamp('2026-10-01')

def main():
    shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
    if os.path.exists(DB): os.remove(DB)
    rows = 0
    for d in range(DAYS):
        frames = []
        for trip, (h, dur) in enumerate([(8, 2700), (18, 2700)]):
            n = dur * HZ
            t = START + pd.Timedelta(days=d, hours=h) + pd.to_timedelta(np.arange(n) / HZ, unit='s')
            hwy = (trip == 1 and d % 3 == 0)
            base, spread = (85, 9) if hwy else (32, 11)
            k = np.ones(100) / 100
            sm = np.convolve(np.convolve(rng.normal(0, 1, n + 200), k, 'valid'), k, 'valid')[:n]
            speed = np.clip(base + spread * sm / sm.std(), 0, 130)
            for _ in range(rng.integers(0, 3)):                       # occasional harsh braking
                s0 = rng.integers(300, n - 300); drop = np.minimum(speed[s0:s0 + 15], np.linspace(0, 28, 15))
                speed[s0:s0 + 15] -= drop; speed[s0 + 15:s0 + 115] -= drop[-1] * np.linspace(1, 0, 100)
            v = speed / 3.6
            accel = np.gradient(v) * HZ
            amb = 24 + 2 * np.sin(d / 5) + (8 * min(1, (d - 27) / 6) if d >= 28 else 0)   # heat wave, last 2 weeks
            tyre = 2.4 - (0.012 * max(0, d - 25))                     # slow leak from day 25
            hvac = 0.6 + 0.18 * max(0, amb - 22) + rng.normal(0, 0.05, n)
            crr = 0.010 * (1 + 0.8 * (2.4 - tyre))
            f_res = 1700 * 9.81 * crr + 0.5 * 1.2 * 0.65 * v**2
            p_in = 1700 * accel * v
            power = (f_res * v / 0.88 + np.where(p_in > 0, p_in / 0.88, p_in * 0.6)) / 1000 + hvac   # kW
            heat = np.convolve(np.abs(power), np.ones(600) / 600, 'same')
            btemp = amb + 5 + 0.35 * heat + rng.normal(0, 0.1, n)
            coolant_flow = np.full(n, 8.0) + rng.normal(0, 0.1, n)
            if d == 36 and trip == 1:                                 # cooling fault -> overheat
                k = n // 2
                coolant_flow[k:] = np.clip(8 - np.linspace(0, 9, n - k), 0.5, None)
                btemp[k:] += np.linspace(0, 24, n - k)
            soc = np.clip(90 - np.cumsum(power) / HZ / 3600 / 40 * 100, 5, 100)
            adas = np.ones(n, bool)
            for _ in range(rng.integers(1, 4)):                       # 1-3 disengagements per trip
                s = rng.integers(600, n - 600); adas[s:s + rng.integers(50, 200)] = False
            frames.append(pd.DataFrame(dict(
                ts=t, trip_id=d * 2 + trip, road=('highway' if hwy else 'city'),
                speed_kph=speed.astype('float32'), accel_ms2=accel.astype('float32'),
                power_kw=power.astype('float32'), soc_pct=soc.astype('float32'),
                batt_temp_c=btemp.astype('float32'), amb_temp_c=np.float32(amb),
                hvac_kw=hvac.astype('float32'), tyre_bar=np.float32(tyre),
                coolant_lpm=coolant_flow.astype('float32'),
                motor_rpm=(speed * 78).astype('float32'), brake_pct=np.clip(-accel * 20, 0, 100).astype('float32'),
                steer_deg=rng.normal(0, 8, n).astype('float32'), adas_on=adas,
                pack_v=(350 + soc * 0.6 + rng.normal(0, 0.3, n)).astype('float32'),
                pack_a=(power * 1000 / 380).astype('float32'),
                cell_min_v=(3.5 + soc * 0.006).astype('float32'), cell_max_v=(3.52 + soc * 0.006).astype('float32'),
                coolant_c=(btemp - 3).astype('float32'), odo_km=np.cumsum(speed / 3.6 / HZ / 1000).astype('float32'),
                lane_conf=rng.random(n).astype('float32'), thr_pct=np.clip(accel * 25 + 20, 0, 100).astype('float32'),
                dtc=np.where(btemp > 55, 'P0A7E', ''))))
        df = pd.concat(frames); rows += len(df)
        duckdb.sql(f"COPY df TO '{OUT}/day={d:02d}.parquet' (FORMAT parquet, COMPRESSION zstd)")
    con = duckdb.connect(DB); con.execute("PRAGMA threads=2; PRAGMA memory_limit='256MB'")
    con.execute(f"CREATE VIEW raw AS SELECT * FROM read_parquet('{OUT}/*.parquet')")
    con.execute("""CREATE TABLE agg1s AS SELECT time_bucket(INTERVAL 1 SECOND, ts) ts, any_value(trip_id) trip_id,
        any_value(road) road, avg(speed_kph) speed_kph, min(accel_ms2) accel_min, avg(power_kw) power_kw, avg(soc_pct) soc_pct,
        max(batt_temp_c) batt_temp_c, avg(amb_temp_c) amb_temp_c, avg(hvac_kw) hvac_kw, avg(tyre_bar) tyre_bar,
        avg(coolant_lpm) coolant_lpm, bool_and(adas_on) adas_on, max(dtc) dtc FROM raw GROUP BY 1""")
    con.execute("""CREATE TABLE trips AS SELECT trip_id, min(ts) t0, any_value(road) road, avg(speed_kph) speed_kph,
        sum(power_kw)/3600 kwh, sum(speed_kph)/3600 km, sum(power_kw)/sum(speed_kph)*100 energy_per_km,
        avg(amb_temp_c) amb_temp_c, avg(hvac_kw) hvac_kw, avg(tyre_bar) tyre_bar, max(batt_temp_c) batt_temp_max,
        sum((accel_min < -3.5)::INT)/(sum(speed_kph)/3600)*100 harsh_per_100km FROM agg1s GROUP BY 1""")
    con.execute("""CREATE TABLE events AS
        SELECT ts, 'adas_disengage' type, 'info' severity, 'speed '||round(speed_kph)::INT||' km/h' detail FROM
          (SELECT *, lag(adas_on) OVER (ORDER BY ts) p FROM agg1s) WHERE p AND NOT adas_on AND ts::TIME > TIME '00:00:01'
        UNION ALL SELECT min(ts), 'battery_overheat', 'critical', 'DTC P0A7E' FROM agg1s WHERE dtc<>'' GROUP BY trip_id
        UNION ALL SELECT ts, 'harsh_brake', 'warn', 'decel '||round(accel_min,1) FROM agg1s WHERE accel_min < -3.5 QUALIFY ts - lag(ts, 1, TIMESTAMP '2000-01-01') OVER (ORDER BY ts) > INTERVAL 5 SECOND""")
    con.execute("CHECKPOINT"); con.close()
    raw = sum(os.path.getsize(f'{OUT}/{f}') for f in os.listdir(OUT))
    print(f"rows={rows} days={DAYS} signals=24 raw_parquet_MB={raw/1e6:.1f} summary_db_MB={os.path.getsize(DB)/1e6:.1f}")

if __name__ == '__main__':
    main()
