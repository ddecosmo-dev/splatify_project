"""Run from PyCharm. Uses the repository's ABODataset; no pycolmap required.
One fresh directory per attempt; never reuse cached outputs for timing.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def command(exe, args):
    exe = str(exe)
    args = list(map(str, args))

    if os.name == "nt" and Path(exe).suffix.lower() in (".bat", ".cmd"):
        exe = os.path.normpath(exe)
        comspec = os.environ.get("COMSPEC", "cmd.exe")
        arguments = subprocess.list2cmdline(args)
        return f'"{comspec}" /d /s /c ""{exe}" {arguments}"'

    return [exe] + args


def kill(proc):
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'], capture_output=True)
    else:
        import signal
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name != 'nt':
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def run(exe, args, log, timeout):
    with log.open('w', encoding='utf-8') as out:
        out.write(json.dumps([exe] + list(map(str, args))) + '\n'); out.flush()
        proc = subprocess.Popen(command(exe, args), stdout=out, stderr=subprocess.STDOUT,
                                start_new_session=os.name != 'nt')
        try:
            rc = proc.wait(timeout=timeout)
        except BaseException:
            kill(proc)
            raise
        if rc:
            raise RuntimeError(f'Exit {rc}; see {log}')


def count_bin(path):
    with path.open('rb') as f:
        return struct.unpack('<Q', f.read(8))[0]


def ply_counts(path):
    counts = {}
    with path.open('rb') as f:
        if f.readline().strip() != b'ply':
            raise ValueError(f'Invalid PLY: {path}')
        for _ in range(200):
            line = f.readline().decode('ascii', errors='replace').strip()
            if line.startswith('element '):
                _, name, n = line.split(); counts[name] = int(n)
            if line == 'end_header':
                return counts
    raise ValueError(f'Invalid PLY header: {path}')


def main():
    from utils.ablation_dataset import ABODataset
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name('config.json')
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    def resolve(p):
        p = Path(p); return (ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    data = resolve(cfg['data_dir']); output = resolve(cfg['output_dir'])
    ds = ABODataset(data_dir=data); products = ds.list_products()
    if len(products) != cfg['expected_products']:
        raise ValueError(f'Expected {cfg["expected_products"]} products, found {len(products)}. Check the curated dataset, do not silently take first 20.')
    if len({p['spin_id'] for p in products}) != len(products):
        raise ValueError('Duplicate spin_id in manifest')
    for p in products:
        for key in ('spin_id', 'product_type'):
            if not re.fullmatch(r'[A-Za-z0-9_-]+', p[key]):
                raise ValueError(f'Unsafe folder identifier: {p[key]}')
        ds.get_image_paths(p['spin_id'], 72)
        ds.get_mesh_path(p['spin_id'])
    if cfg['mode'] not in ('check', 'pilot', 'all'):
        raise ValueError('mode must be check, pilot or all')
    if cfg['mode'] == 'pilot':
        pid = cfg.get('pilot_product_id')
        products = [p for p in products if pid in (p['spin_id'], p['item_id'])] if pid else products[:1]
        if len(products) != 1: raise ValueError('pilot_product_id must identify one product')
    counts = cfg['image_counts']
    if sorted(counts) != [18, 36, 72]: raise ValueError('Use exactly 18, 36, 72')
    output.mkdir(parents=True, exist_ok=True)
    exe = cfg['colmap_executable']
    if not Path(exe).is_file():
        exe = shutil.which(exe)
    if not exe: raise FileNotFoundError('Set colmap_executable to your COLMAP.bat or colmap executable')
    help_text = {}
    for cmd in ('feature_extractor', 'exhaustive_matcher', 'mapper', 'image_undistorter',
                'patch_match_stereo', 'stereo_fusion', 'poisson_mesher'):
        r = subprocess.run(command(exe, [cmd, '-h']), capture_output=True, timeout=40)
        txt = (r.stdout + r.stderr).decode('utf-8', errors='replace')
        if '--' not in txt: raise RuntimeError(f'Cannot load help for {cmd}: {txt[-1500:]}')
        help_text[cmd] = txt
    def flag(cmd, *choices):
        for choice in choices:
            if '--' + choice in help_text[cmd]: return '--' + choice
        raise RuntimeError(f'Unsupported COLMAP option: {choices}')
    ext_gpu = flag('feature_extractor', 'FeatureExtraction.use_gpu', 'SiftExtraction.use_gpu')
    mat_gpu = flag('exhaustive_matcher', 'FeatureMatching.use_gpu', 'SiftMatching.use_gpu')
    # Save actual binary help/defaults and manifest fingerprint for provenance.
    fingerprint = hashlib.sha256((data/'selected_products.csv').read_bytes()).hexdigest()
    protocol = {k: v for k, v in cfg.items() if k not in ('mode','pilot_product_id')}
    protocol.update(manifest_sha256=fingerprint, colmap_help=help_text)

    def normalize_protocol(value):
        result = dict(value)
        result["colmap_help"] = {
            name: re.sub(
                r"^[IWEF]\d{8}\s+\d{2}:\d{2}:\d{2}\.\d+\s+\d+\s+"
                r"[^ \r\n]+:\d+\]\s?",
                "",
                text,
                flags=re.MULTILINE,
            )
            for name, text in value.get("colmap_help", {}).items()
        }
        return result

    normalized = normalize_protocol(protocol)
    signature = hashlib.sha256(
        json.dumps(normalized, sort_keys=True).encode()
    ).hexdigest()

    protocol_path = output / "protocol.json"

    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
        previous = normalize_protocol(old["protocol"])

        if previous != normalized:
            changed = [
                key for key in sorted(set(previous) | set(normalized))
                if previous.get(key) != normalized.get(key)
            ]
            raise ValueError(
                f"Actual protocol changes: {changed}. "
                "Choose a NEW output_dir for a new benchmark."
            )

    save(protocol_path, dict(signature=signature, protocol=protocol))
    print(f'Dataset OK. {len(products)} objects x 3 counts. Mode={cfg["mode"]}', flush=True)
    if cfg['mode'] == 'check':
        print('Check only: no reconstruction launched. CUDA availability still requires a pilot run.'); return
    # Prevent accidentally starting two writers/GPU batches with the same output.
    lock = output/'RUNNING.lock'
    with lock.open('x') as f: f.write(str(os.getpid()))
    try:
        for product in products:
            for n in counts:
                base = output/product['spin_id']/str(n); base.mkdir(parents=True, exist_ok=True)
                prior = sorted(base.glob('attempt_*/result.json'))
                if prior:
                    prev = json.loads(prior[-1].read_text(encoding="utf-8"))
                    if prev['status'] in ('completed_full_registration','completed_partial_registration'):
                        print(f'Skip completed: {base}', flush=True); continue
                attempt = base/f'attempt_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
                attempt.mkdir(); images=attempt/'images'; images.mkdir()
                sparse=attempt/'sparse'; sparse.mkdir(); dense=attempt/'dense'; dense.mkdir()
                record = dict(product, input_images=n, status='running', failed_stage='', error='',
                              registered_images=None, sparse_points=None, fused_points=None,
                              mesh_vertices=None, mesh_faces=None, attempt=str(attempt),
                              geometry_metric=None, signature=signature, stage_seconds={})
                started = time.perf_counter(); stage='prepare_inputs'
                def step(name, cmd, opts):
                    nonlocal stage
                    stage=name
                    print(f'{product["spin_id"]} / {n}: {name}', flush=True)
                    t=time.perf_counter()
                    try: run(exe, [cmd]+opts, attempt/f'{name}.log', cfg['stage_timeout_seconds'])
                    finally:
                        record['stage_seconds'][name]=round(time.perf_counter()-t,3)
                        save(attempt/'result.json', record)
                try:
                    selected=ds.get_image_paths(product['spin_id'], n)
                    save(attempt/'selected_images.json', [str(p) for p in selected])
                    for p in selected: shutil.copy2(p,images/p.name)
                    db=attempt/'database.db'
                    step('features','feature_extractor',['--database_path',db,'--image_path',images,
                        '--ImageReader.camera_model','SIMPLE_RADIAL','--ImageReader.single_camera',cfg['single_camera'],
                        '--SiftExtraction.max_num_features',8192,ext_gpu,1])
                    step('matching','exhaustive_matcher',['--database_path',db,mat_gpu,1])
                    step('sfm','mapper',['--database_path',db,'--image_path',images,'--output_path',sparse])
                    models=[p for p in sparse.iterdir() if p.is_dir() and (p/'images.bin').exists() and (p/'points3D.bin').exists()]
                    if not models: raise RuntimeError('No sparse model; mapper produced no valid initial reconstruction')
                    best=max(models,key=lambda p:(count_bin(p/'images.bin'),count_bin(p/'points3D.bin')))
                    record.update(registered_images=count_bin(best/'images.bin'), sparse_points=count_bin(best/'points3D.bin'), sparse_model=str(best), sparse_components=len(models))
                    if record['registered_images']<3 or record['sparse_points']==0:
                        raise RuntimeError('Fewer than 3 registered views or no points; dense stages skipped by protocol')
                    step('undistortion','image_undistorter',['--image_path',images,'--input_path',best,'--output_path',dense,'--output_type','COLMAP','--max_image_size',cfg['max_image_size']])
                    step('stereo','patch_match_stereo',['--workspace_path',dense,'--workspace_format','COLMAP','--PatchMatchStereo.geom_consistency','true','--PatchMatchStereo.max_image_size',cfg['max_image_size']])
                    fused=dense/'fused.ply'; mesh=dense/'meshed-poisson.ply'
                    step('fusion','stereo_fusion',['--workspace_path',dense,'--workspace_format','COLMAP','--input_type','geometric','--output_path',fused])
                    record['fused_points']=ply_counts(fused).get('vertex',0)
                    if not record['fused_points']: raise RuntimeError('Empty fused cloud')
                    record['fused_path']=str(fused)
                    if cfg['meshing']:
                        step('meshing','poisson_mesher',['--input_path',fused,'--output_path',mesh,'--PoissonMeshing.depth',cfg['poisson_depth'],'--PoissonMeshing.point_weight',1,'--PoissonMeshing.trim',cfg['poisson_trim']])
                        pc=ply_counts(mesh);record.update(mesh_vertices=pc.get('vertex',0),mesh_faces=pc.get('face',0),mesh_path=str(mesh))
                        if not record['mesh_faces']: raise RuntimeError('Empty mesh')
                    record['status']='completed_full_registration' if record['registered_images']==n else 'completed_partial_registration'
                except KeyboardInterrupt:
                    record.update(status='interrupted',failed_stage=stage,error='User interrupted'); raise
                except Exception as e:
                    record.update(status='failed',failed_stage=stage,error=str(e));print(str(e),flush=True)
                finally:
                    record['total_seconds']=round(time.perf_counter()-started,3)
                    save(attempt/'result.json',record)
                    summary(output)
    finally:
        lock.unlink(missing_ok=True)


def summary(output):
    rows=[]
    for base in sorted(output.glob('*/*')):
        paths=sorted(base.glob('attempt_*/result.json')) if base.is_dir() else []
        if not paths: continue
        row=json.loads(paths[-1].read_text(encoding="utf-8"))
        times=row.pop('stage_seconds',{})
        row.update({k+'_seconds':v for k,v in times.items()});rows.append(row)
    if rows:
        keys=sorted(set().union(*(r.keys() for r in rows)))
        tmp=output/'summary.tmp'
        with tmp.open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
        tmp.replace(output/'summary.csv')


if __name__=='__main__':
    main()
