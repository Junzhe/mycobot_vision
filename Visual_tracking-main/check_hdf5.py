import argparse, sys, os, glob
import numpy as np
import h5py

OK   = 0
WARN = 1
FAIL = 2

# ---------------------------- 工具函数 ----------------------------

def pct(x, n): 
    return 0.0 if n == 0 else 100.0 * x / n

def is_rotation_matrix(Rm, tol_orth=1e-3, tol_det=1e-2):
    R = Rm[:3,:3]
    err = np.linalg.norm(R.T @ R - np.eye(3), ord='fro')
    d = np.linalg.det(R)
    return (err < tol_orth) and (abs(d-1.0) < tol_det), err, d

def safe_get(g, name, default=None):
    try:
        if name in g:
            return g[name][...]
    except Exception:
        pass
    return default

# ---------------------------- 各项检查 ----------------------------

def check_attrs(f):
    status, msgs = OK, []
    need = ["st_pos_unit","st_rot_unit","ac_pos_unit","ac_rot_unit"]
    for k in need:
        if k not in f.attrs:
            status = max(status, WARN)
            msgs.append(f"[WARN] file.attrs 缺少 {k}")

    def _read_attr(key):
        v = f.attrs.get(key, "")
        return v.decode() if isinstance(v, (bytes, bytearray)) else v

    up = _read_attr("st_pos_unit")
    ur = _read_attr("st_rot_unit")
    if up and up not in ("mm",):
        status = max(status, WARN); msgs.append(f"[WARN] st_pos_unit={up} (期望 mm)")
    if ur and ur not in ("deg",):
        status = max(status, WARN); msgs.append(f"[WARN] st_rot_unit={ur} (期望 deg)")

    T_ee_cam = f.attrs.get("T_ee_cam_mm", None)
    if T_ee_cam is None:
        msgs.append("[WARN] 未找到 attrs['T_ee_cam_mm']，几何一致性仅作近似校验")
        status = max(status, WARN)
        T_ee_cam = np.eye(4, dtype=np.float32)
    else:
        T_ee_cam = np.array(T_ee_cam, dtype=float).reshape(4,4)
        okR, err, det = is_rotation_matrix(T_ee_cam, 1e-3, 5e-2)
        if not okR:
            status = max(status, WARN)
            msgs.append(f"[WARN] T_ee_cam 旋转不正交或 det!=1 (fro_err={err:.3e}, det={det:.4f})")
    return status, msgs, T_ee_cam

def check_basic_shapes(fr):
    status, msgs = OK, []

    # 基本需要的主序列（允许 target/visible 缺失但会降级提示）
    required = [
        "images/rgb", "images/mask",
        "state",
        "action_cam", "action_base",
        "vel_cam", "vel_base",
        "time/ts", "time/dt",
        "joints/angles_deg", "joints/vel_deg_s",
        "poses/T_be_mm", "poses/T_bc_mm",
        "cmd/type", "cmd/speed", "cmd/coords_mmdeg", "cmd/angles_deg", "cmd/gripper", "cmd/time",
        "cond/target", "cond/phase",
        "target/cam_mm", "target/base_mm"
    ]
    missing = [k for k in required if k not in fr]
    if missing:
        status = max(status, FAIL)
        msgs.append(f"[FAIL] 缺少数据集: {missing}")

    # target/visible 建议有
    if "target/visible" not in fr:
        status = max(status, WARN)
        msgs.append("[WARN] 建议提供 target/visible（0/1 可见标记）以更好判断 NaN 合理性")

    # 形状
    def shp(name): return fr[name].shape if name in fr else None

    if "images/rgb" in fr and "images/mask" in fr:
        s_rgb, s_msk = shp("images/rgb"), shp("images/mask")
        if not (isinstance(s_rgb, tuple) and len(s_rgb) == 4 and s_rgb[-1] == 3):
            status = max(status, FAIL); msgs.append(f"[FAIL] images/rgb 形状异常: {s_rgb} (期望 [N,H,W,3])")
        if not (isinstance(s_msk, tuple) and len(s_msk) == 3):
            status = max(status, FAIL); msgs.append(f"[FAIL] images/mask 形状异常: {s_msk} (期望 [N,H,W])")
        if isinstance(s_rgb, tuple) and isinstance(s_msk, tuple):
            if s_rgb[0] != s_msk[0] or s_rgb[1] != s_msk[1] or s_rgb[2] != s_msk[2]:
                status = max(status, FAIL)
                msgs.append(f"[FAIL] 图像与掩码维度不一致: rgb={s_rgb}, mask={s_msk}")

    # 主序列长度一致性
    names_main_seq = [
        "state","action_cam","action_base","vel_cam","vel_base",
        "time/ts","time/dt","joints/angles_deg","joints/vel_deg_s",
        "poses/T_be_mm","poses/T_bc_mm",
        "cmd/type","cmd/speed","cmd/coords_mmdeg","cmd/angles_deg","cmd/gripper","cmd/time",
        "cond/target","cond/phase",
        "target/cam_mm","target/base_mm"
    ]
    lens = [fr[n].shape[0] for n in names_main_seq if n in fr]
    if lens:
        if len(set(lens)) != 1:
            status = max(status, FAIL)
            msgs.append(f"[FAIL] 主序列长度不一致: { {n: fr[n].shape for n in names_main_seq if n in fr} }")
    else:
        status = max(status, FAIL)
        msgs.append("[FAIL] 未找到任何主序列数据集")

    return status, msgs

def check_time(fr):
    status, msgs = OK, []
    ts, dt = safe_get(fr, "time/ts"), safe_get(fr, "time/dt")
    if ts is None or dt is None:
        return FAIL, ["[FAIL] 缺少 time/ts 或 time/dt"]
    if ts.ndim != 1 or dt.ndim != 1 or len(ts) != len(dt):
        return FAIL, [f"[FAIL] ts/dt 形状异常: ts={ts.shape}, dt={dt.shape}"]

    diff_ts = np.diff(ts)
    n_nonmono = int(np.sum(diff_ts <= 0))
    if n_nonmono > 0:
        status = max(status, FAIL); msgs.append(f"[FAIL] ts 非严格递增: {n_nonmono}/{len(diff_ts)} 处不递增")

    dt_from_ts = np.r_[diff_ts, diff_ts[-1] if len(diff_ts) else 0]
    bad = np.abs(dt_from_ts - dt) > 0.003
    n_bad = int(np.sum(bad))
    if n_bad > 0:
        status = max(status, WARN); msgs.append(f"[WARN] dt 与 ts 差分不一致: {n_bad}/{len(dt)} 处 > 3ms")

    out = (dt < 0.002) | (dt > 0.2)
    n_out = int(np.sum(out))
    if n_out > 0:
        status = max(status, WARN); msgs.append(f"[WARN] dt 超常: {n_out}/{len(dt)} ∉ [2ms,200ms]")
    return status, msgs

def check_images(fr):
    status, msgs = OK, []
    rgb, msk = safe_get(fr, "images/rgb"), safe_get(fr, "images/mask")
    if rgb is None or msk is None:
        return WARN, ["[WARN] 无图像/掩码，跳过图像检查"]
    N,H,W,C = rgb.shape
    if C != 3:
        status = max(status, FAIL); msgs.append(f"[FAIL] RGB 通道数={C} (应为 3)")
    samp = rgb.reshape(-1,3)[::max(1,(N*H*W)//20000)]
    if (samp.min() < 0) or (samp.max() > 255):
        status = max(status, FAIL); msgs.append(f"[FAIL] RGB 值域异常: min={samp.min()}, max={samp.max()} (应在[0,255])")
    mu = np.unique(msk.reshape(-1)[::max(1,(N*H*W)//50000)])
    bad = [v for v in mu if v not in (0,255)]
    if bad:
        status = max(status, WARN); msgs.append(f"[WARN] 掩码存在非 0/255 值: {bad[:10]}")
    return status, msgs

def check_joints(fr):
    status, msgs = OK, []
    q, dq, dt = safe_get(fr,"joints/angles_deg"), safe_get(fr,"joints/vel_deg_s"), safe_get(fr,"time/dt")
    if q is None or dq is None or dt is None:
        return WARN, ["[WARN] 关节/速度/时间缺失，跳过关节检查"]
    if q.ndim != 2 or q.shape[1] != 6:
        return FAIL, [f"[FAIL] joints/angles_deg 形状异常: {q.shape}"]
    if dq.shape != q.shape:
        return FAIL, [f"[FAIL] joints/vel_deg_s 形状异常: {dq.shape}"]

    dtc = np.clip(dt, 1e-3, None)
    re_dq = np.zeros_like(q)
    re_dq[1:] = (q[1:] - q[:-1]) / dtc[1:,None]
    mae = np.nanmedian(np.abs(dq - re_dq))
    if mae > 5.0:
        status = max(status, WARN); msgs.append(f"[WARN] 关节速度与(q差/dt)不一致: median|Δ|={mae:.2f} deg/s (>5)")
    out = (q < -3600) | (q > 3600)
    n_out = int(np.sum(out))
    if n_out > 0:
        status = max(status, WARN); msgs.append(f"[WARN] 极端关节角 {n_out} 个样本疑似异常 (>|3600| deg)")
    return status, msgs

def check_transforms(fr):
    status, msgs = OK, []
    T_be, T_bc = safe_get(fr,"poses/T_be_mm"), safe_get(fr,"poses/T_bc_mm")
    if T_be is None or T_bc is None:
        return WARN, ["[WARN] 缺少 T_be/T_bc，跳过 SE(3) 检查"]
    bad_be = sum(not is_rotation_matrix(T, 5e-3, 5e-2)[0] for T in T_be)
    bad_bc = sum(not is_rotation_matrix(T, 5e-3, 5e-2)[0] for T in T_bc)
    if bad_be or bad_bc:
        status = max(status, WARN)
        msgs.append(f"[WARN] 非正交旋转: T_be {bad_be}/{len(T_be)}, T_bc {bad_bc}/{len(T_bc)}")
    return status, msgs

def check_actions_vels(fr):
    status, msgs = OK, []
    ac_cam, ac_base = safe_get(fr,"action_cam"), safe_get(fr,"action_base")
    vv_cam, vv_base = safe_get(fr,"vel_cam"),    safe_get(fr,"vel_base")
    dt = safe_get(fr,"time/dt")
    if ac_cam is None or vv_cam is None or dt is None:
        return WARN, ["[WARN] 缺少 action/vel/dt，跳过动作速度检查"]

    dtc = np.clip(dt, 1e-3, None)
    pred_v_cam  = np.zeros_like(vv_cam)
    pred_v_cam[:, :6] = ac_cam[:, :6] / dtc[:, None]
    err_cam = np.nanmedian(np.abs(pred_v_cam[:,:6] - vv_cam[:,:6]), axis=0)
    if np.any(err_cam[:3] > 10.0) or np.any(err_cam[3:6] > 10.0):
        status = max(status, WARN)
        msgs.append(f"[WARN] vel_cam 与 action_cam/dt 不一致: median|Δ|lin={err_cam[:3]}, ang={err_cam[3:6]}")

    if (ac_base is not None) and (vv_base is not None):
        pred_v_base = np.zeros_like(vv_base)
        pred_v_base[:, :6] = ac_base[:, :6] / dtc[:, None]
        err_base = np.nanmedian(np.abs(pred_v_base[:,:6] - vv_base[:,:6]), axis=0)
        if np.any(err_base[:3] > 10.0) or np.any(err_base[3:6] > 10.0):
            status = max(status, WARN)
            msgs.append(f"[WARN] vel_base 与 action_base/dt 不一致: median|Δ|lin={err_base[:3]}, ang={err_base[3:6]}")

        # 线速度过大提示（95 分位）
        sp = np.linalg.norm(vv_base[:,:3], axis=1)
        p95 = float(np.nanpercentile(sp, 95))
        if p95 > 600.0:
            status = max(status, WARN)
            msgs.append(f"[WARN] vel_base 线速度>600mm/s，95分位≈{p95:.1f}，疑似异常或单帧抖动")
    return status, msgs

def check_target_visibility_nan(fr):
    """用 target/visible 来判定 cam_mm 的 NaN 是否合理（抓取后遮挡是预期）"""
    status, msgs = OK, []
    t_cam = safe_get(fr, "target/cam_mm")
    t_vis = safe_get(fr, "target/visible")
    if t_cam is None:
        return WARN, ["[WARN] 缺少 target/cam_mm，跳过可见性一致性检查"]
    a = np.asarray(t_cam, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3:
        return WARN, [f"[WARN] target/cam_mm 形状异常: {a.shape}，跳过可见性检查"]

    nan_rate = float(np.isnan(a).any(axis=1).mean())
    if t_vis is None:
        # 没有 visible，只给出提示，不做 FAIL
        status = max(status, WARN)
        msgs.append(f"[WARN] 未提供 target/visible，无法判断 NaN 是否合理；当前 cam_mm NaN≈{nan_rate*100:.1f}%")
        return status, msgs

    vis = np.asarray(t_vis).astype(np.uint8).reshape(-1)
    if vis.ndim != 1 or len(vis) != len(a):
        return WARN, ["[WARN] target/visible 形状异常，跳过可见性检查"]

    vis_rate = float((vis > 0).mean())
    exp_nan_rate = 1.0 - vis_rate
    if abs(nan_rate - exp_nan_rate) <= 0.10:  # 10% 容差
        msgs.append(f"[INFO] target/cam_mm NaN≈{nan_rate*100:.1f}% 与不可见率≈{exp_nan_rate*100:.1f}% 一致（抓取后遮挡属预期）")
    else:
        status = max(status, WARN)
        msgs.append(f"[WARN] target/cam_mm NaN={nan_rate*100:.1f}% 与不可见率={exp_nan_rate*100:.1f}% 偏差较大，请排查")
    return status, msgs

def check_targets_geometry(fr, T_ee_cam):
    status, msgs = OK, []
    T_be = safe_get(fr, "poses/T_be_mm")
    t_cam = safe_get(fr, "target/cam_mm")
    t_base= safe_get(fr, "target/base_mm")
    if T_be is None or t_cam is None or t_base is None:
        return WARN, ["[WARN] 缺少目标或位姿，跳过 target 几何一致性"]
    bad, tot = 0, 0
    for i in range(len(t_cam)):
        tc = t_cam[i]; tb = t_base[i]
        if np.any(np.isnan(tc)) or np.any(np.isnan(tb)):  # cam 不可见或锁定为夹爪中心时跳过
            continue
        p_cam_h = np.r_[tc.astype(float), 1.0]
        pred_b = (T_be[i] @ T_ee_cam @ p_cam_h)[:3]
        e = np.linalg.norm(pred_b - tb.astype(float))
        if e > 30.0:  # 3cm
            bad += 1
        tot += 1
    if tot > 0 and bad / tot > 0.1:
        status = max(status, WARN)
        msgs.append(f"[WARN] 目标几何一致性较差: >3cm 比率 {pct(bad, tot):.1f}% ({bad}/{tot})")
    elif tot == 0:
        msgs.append("[INFO] 目标在大多数帧不可见（或已锁定为夹爪），未做 cam→base 一致性统计")
    return status, msgs

def check_cmd(fr):
    status, msgs = OK, []
    ctype = safe_get(fr, "cmd/type")
    if ctype is None:
        return WARN, ["[WARN] 缺少 cmd/*，跳过真实命令检查"]

    coords = safe_get(fr, "cmd/coords_mmdeg")
    angles = safe_get(fr, "cmd/angles_deg")
    grip   = safe_get(fr, "cmd/gripper")
    speed  = safe_get(fr, "cmd/speed")

    N = len(ctype)
    valid = 0
    kinds = set()
    for i in range(N):
        if int(ctype[i]) != 0:
            valid += 1; kinds.add(int(ctype[i])); continue
        cc = coords[i] if coords is not None else np.full(6, np.nan)
        aa = angles[i] if angles is not None else np.full(6, np.nan)
        gg = int(grip[i]) if grip is not None else -1
        if (not np.all(np.isnan(cc))) or (not np.all(np.isnan(aa))) or (gg in (0,1)):
            valid += 1
    cov = pct(valid, N)
    sp_mean = float(np.nanmean(speed)) if speed is not None else np.nan
    msgs.append(f"[INFO] cmd 覆盖率: {cov:.1f}%  种类: {sorted(kinds) if kinds else '[]'}  平均速度: {sp_mean}")
    if cov < 50.0:
        status = max(status, WARN); msgs.append("[WARN] 真实命令覆盖率偏低，建议通过 HTTP 封装统一下发以便记录")
    return status, msgs

def check_phase(fr):
    status, msgs = OK, []
    ph = safe_get(fr, "cond/phase")
    if ph is None:
        return WARN, ["[WARN] 缺少 cond/phase"]
    uniq = sorted(set(int(x) for x in ph.reshape(-1)))
    allow = {0,1,2,3}  # 适配 confirmed=3
    bad = [x for x in uniq if x not in allow]
    if bad:
        status = max(status, WARN)
        msgs.append(f"[WARN] cond/phase 出现非 {sorted(allow)} 值: {bad}")
    else:
        msgs.append(f"[INFO] 夹爪/相位近况: 相位集合={uniq}")
    return status, msgs

def summarize_visibility(fr):
    msgs = []
    vis = safe_get(fr, "target/visible")
    if vis is None:
        return msgs
    vis_rate = float((vis.reshape(-1) > 0).mean())
    msgs.append(f"[INFO] target 可见率: {vis_rate*100:.1f}%")
    # 抓取前长时间不可见（例如靠近遮挡）
    ph = safe_get(fr, "cond/phase")
    first_vis = int(np.argmax(vis > 0)) if np.any(vis > 0) else len(vis)
    if first_vis > 30:  # > 30 帧
        msgs.append(f"[WARN] 抓取前存在较长不可见段（长度≈{first_vis} 帧）")
    return msgs

def summarize_nan_outliers(fr):
    """通用 NaN 汇总（跳过 target/cam_mm，由可见性专检）"""
    status, msgs = OK, []
    keys = [
        "state","action_cam","action_base","vel_cam","vel_base",
        "joints/angles_deg","joints/vel_deg_s",
        "poses/T_be_mm","poses/T_bc_mm",
        # "target/cam_mm",  # 专检
        "target/base_mm"
    ]
    for k in keys:
        if k not in fr:  # 形状检查会报
            continue
        arr = np.asarray(fr[k][...], dtype=float)
        n_nan = int(np.isnan(arr).sum())
        if n_nan > 0:
            rate = pct(n_nan, arr.size)
            lvl = WARN if rate < 5.0 else FAIL
            status = max(status, lvl)
            msgs.append(f"[{'WARN' if lvl==WARN else 'FAIL'}] {k} 含 NaN: {rate:.2f}%")
    return status, msgs

# ---------------------------- 主流程 ----------------------------

def check_file(path):
    st = OK
    report = [f"=== {os.path.basename(path)} ==="]
    try:
        with h5py.File(path, "r") as f:
            if "frames" not in f:
                report.append("[FAIL] 根下无 'frames' 组")
                return FAIL, "\n".join(report)
            fr = f["frames"]

            s, m, T_ee = check_attrs(f);               st=max(st,s); report+=m
            s, m = check_basic_shapes(fr);             st=max(st,s); report+=m
            s, m = check_time(fr);                     st=max(st,s); report+=m
            s, m = check_images(fr);                   st=max(st,s); report+=m
            s, m = check_joints(fr);                   st=max(st,s); report+=m
            s, m = check_transforms(fr);               st=max(st,s); report+=m
            s, m = check_actions_vels(fr);             st=max(st,s); report+=m
            s, m = check_phase(fr);                    st=max(st,s); report+=m
            # 可见性一致性 + 几何一致性（可见帧）
            s, m = check_target_visibility_nan(fr);    st=max(st,s); report+=m
            s, m = check_targets_geometry(fr, T_ee);   st=max(st,s); report+=m
            s, m = check_cmd(fr);                      st=max(st,s); report+=m
            s, m = summarize_nan_outliers(fr);         st=max(st,s); report+=m
            report += summarize_visibility(fr)

            # 友好总结
            N = fr["state"].shape[0] if "state" in fr else 0
            report.append(f"[INFO] 帧数: {N}")
            if "images/rgb" in fr:
                report.append(f"[INFO] 图像尺寸: {fr['images/rgb'].shape[1:4]}")
            succ = f.attrs.get("success", None); coll = f.attrs.get("collision", None)
            report.append(f"[INFO] success={succ}  collision={coll}")

    except Exception as e:
        st = FAIL
        report.append(f"[FAIL] 读取/解析异常: {e}")
        return st, "\n".join(report)

    # 末尾摘要（训练友好视角）
    if st == OK:
        report.append("[SUMMARY] ✅ 无 FAIL，检查通过，可直接用于训练")
    elif st == WARN:
        report.append("[SUMMARY] ⚠️ 仅 WARN，通常可通过 delta/dt 或重采样在训练时对冲")
    else:
        report.append("[SUMMARY] ❌ 存在 FAIL，请根据上方提示修复后再训练")

    return st, "\n".join(report)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="hdf5 文件或目录")
    ap.add_argument("--glob", default="*.hdf5", help="目录下的通配符 (默认 *.hdf5)")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        if os.path.isdir(p):
            files.extend(glob.glob(os.path.join(p, args.glob)))
        else:
            files.append(p)
    files = sorted(set(files))

    if not files:
        print("请提供待检查的文件或目录，例如：\n  python3 check_hdf5.py /path/to/dir --glob \"*.hdf5\"")
        return 2

    exit_code = OK
    for f in files:
        st, rep = check_file(f)
        print(rep, "\n")
        exit_code = max(exit_code, st)

    return exit_code

if __name__ == "__main__":
    sys.exit(main())
