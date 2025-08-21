# -*- coding: utf-8 -*-
"""
check_hdf5.py — 体检你采集的 HDF5 数据（mm/deg 版本）
依赖: numpy, h5py
用法:
  python check_hdf5.py file1.hdf5 file2.hdf5
  python check_hdf5.py path/to/dir --glob "*.hdf5"
"""
import argparse, sys, os, glob, math
import numpy as np
import h5py

OK   = 0
WARN = 1
FAIL = 2

def pct(x, n): 
    return 0.0 if n == 0 else 100.0 * x / n

def det3(R):
    return np.linalg.det(R[:3,:3])

def is_rotation_matrix(Rm, tol_orth=1e-3, tol_det=1e-2):
    R = Rm[:3,:3]
    should_I = R.T @ R
    err = np.linalg.norm(should_I - np.eye(3), ord='fro')
    d = np.linalg.det(R)
    return (err < tol_orth) and (abs(d-1.0) < tol_det), err, d

def load_required(g, name):
    if name not in g:
        raise KeyError(f"missing dataset: {name}")
    return g[name][...]

def safe_get(g, name, default=None):
    try:
        if name in g: return g[name][...]
    except Exception:
        pass
    return default

def check_attrs(f):
    status = OK
    msgs = []
    need = ["st_pos_unit","st_rot_unit","ac_pos_unit","ac_rot_unit"]
    for k in need:
        if k not in f.attrs:
            status = max(status, WARN)
            msgs.append(f"[WARN] file.attrs 缺少 {k}")
    # 单位预期
    up = f.attrs.get("st_pos_unit", "").decode() if isinstance(f.attrs.get("st_pos_unit", ""), bytes) else f.attrs.get("st_pos_unit", "")
    ur = f.attrs.get("st_rot_unit", "").decode() if isinstance(f.attrs.get("st_rot_unit", ""), bytes) else f.attrs.get("st_rot_unit", "")
    if up and up not in ("mm",):
        status = max(status, WARN); msgs.append(f"[WARN] st_pos_unit={up} (期望 mm)")
    if ur and ur not in ("deg",):
        status = max(status, WARN); msgs.append(f"[WARN] st_rot_unit={ur} (期望 deg)")

    # T_ee_cam
    T_ee_cam = f.attrs.get("T_ee_cam_mm", None)
    if T_ee_cam is None:
        msgs.append("[WARN] 未找到 attrs['T_ee_cam_mm']，将用单位矩阵进行几何一致性近似校验")
        status = max(status, WARN)
        T_ee_cam = np.eye(4, dtype=np.float32)
    else:
        T_ee_cam = np.array(T_ee_cam, dtype=float).reshape(4,4)
        okR, err, det = is_rotation_matrix(T_ee_cam, 1e-3, 5e-2)
        if not okR:
            status = max(status, WARN)
            msgs.append(f"[WARN] T_ee_cam 旋转不正交或 det != 1 (fro_err={err:.3e}, det={det:.4f})")
    return status, msgs, T_ee_cam

def check_basic_shapes(fr):
    status = OK
    msgs = []
    # 必备组
    need = [
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
    missing = [k for k in need if k not in fr]
    if missing:
        status = max(status, FAIL)
        msgs.append(f"[FAIL] 缺少数据集: {missing}")

    def shape(name):
        return fr[name].shape if name in fr else None

    if "images/rgb" in fr and "images/mask" in fr:
        s_rgb  = shape("images/rgb")   # [N,H,W,3]
        s_mask = shape("images/mask")  # [N,H,W]
        if len(s_rgb or []) != 4 or (s_rgb[-1] != 3):
            status = max(status, FAIL); msgs.append(f"[FAIL] images/rgb 形状异常: {s_rgb} (期望 [N,H,W,3])")
        if len(s_mask or []) != 3:
            status = max(status, FAIL); msgs.append(f"[FAIL] images/mask 形状异常: {s_mask} (期望 [N,H,W])")
        if s_rgb and s_mask and (s_rgb[0] != s_mask[0] or s_rgb[1] != s_mask[1] or s_rgb[2] != s_mask[2]):
            status = max(status, FAIL); msgs.append(f"[FAIL] 图像与掩码维度不一致: rgb={s_rgb}, mask={s_mask}")

    # 时间长度一致性（与其他主序列）
    names_main_seq = [
        "state","action_cam","action_base","vel_cam","vel_base",
        "time/ts","time/dt","joints/angles_deg","joints/vel_deg_s",
        "poses/T_be_mm","poses/T_bc_mm",
        "cmd/type","cmd/speed","cmd/coords_mmdeg","cmd/angles_deg","cmd/gripper","cmd/time",
        "cond/target","cond/phase",
        "target/cam_mm","target/base_mm"
    ]
    lens = []
    for n in names_main_seq:
        if n in fr:
            lens.append(fr[n].shape[0])
    if lens:
        uniq = set(lens)
        if len(uniq) != 1:
            status = max(status, FAIL)
            msgs.append(f"[FAIL] 主序列长度不一致: {dict((n, fr[n].shape) for n in names_main_seq if n in fr)}")
    else:
        status = max(status, FAIL)
        msgs.append("[FAIL] 未找到任何主序列数据集")

    return status, msgs

def check_time(fr):
    status = OK; msgs = []
    ts = safe_get(fr, "time/ts"); dt = safe_get(fr, "time/dt")
    if ts is None or dt is None:
        return FAIL, ["[FAIL] 缺少 time/ts 或 time/dt"]

    if ts.ndim != 1 or dt.ndim != 1 or len(ts) != len(dt):
        return FAIL, [f"[FAIL] ts/dt 形状异常: ts={ts.shape}, dt={dt.shape}"]

    # 单调性
    diff_ts = np.diff(ts)
    n_nonmono = int(np.sum(diff_ts <= 0))
    if n_nonmono > 0:
        status = max(status, FAIL); msgs.append(f"[FAIL] ts 非严格递增: {n_nonmono}/{len(diff_ts)} 处不递增")

    # dt 与 ts 差分一致性（允许 3ms 误差）
    dt_from_ts = np.r_[diff_ts, diff_ts[-1] if len(diff_ts) else 0]
    bad = np.abs(dt_from_ts - dt) > 0.003
    n_bad = int(np.sum(bad))
    if n_bad > 0:
        status = max(status, WARN); msgs.append(f"[WARN] dt 与 ts 差分不一致: {n_bad}/{len(dt)} 处 > 3ms")

    # dt 合理范围（2ms ~ 200ms）
    out = (dt < 0.002) | (dt > 0.2)
    n_out = int(np.sum(out))
    if n_out > 0:
        status = max(status, WARN); msgs.append(f"[WARN] dt 非常规采样间隔: {n_out}/{len(dt)} ∉ [2ms, 200ms]")

    return status, msgs

def check_images(fr):
    status = OK; msgs = []
    rgb = safe_get(fr, "images/rgb"); msk = safe_get(fr, "images/mask")
    if rgb is None or msk is None: 
        return WARN, ["[WARN] 无图像/掩码，跳过图像检查"]
    N,H,W,C = rgb.shape
    if C != 3:
        status = max(status, FAIL); msgs.append(f"[FAIL] RGB 通道数={C} (应为3)")
    # 取样检查值域
    samp = rgb.reshape(-1,3)[::max(1,(N*H*W)//20000)]
    if (samp.min() < 0) or (samp.max() > 255):
        status = max(status, FAIL); msgs.append(f"[FAIL] RGB 值域异常: min={samp.min()}, max={samp.max()} (应在[0,255])")
    # 掩码 0/255
    mu = np.unique(msk.reshape(-1)[::max(1,(N*H*W)//50000)])
    bad = [v for v in mu if v not in (0,255)]
    if bad:
        status = max(status, WARN); msgs.append(f"[WARN] 掩码存在非 0/255 值: {bad[:10]}")
    return status, msgs

def check_joints(fr):
    status = OK; msgs = []
    q  = safe_get(fr, "joints/angles_deg")
    dq = safe_get(fr, "joints/vel_deg_s")
    dt = safe_get(fr, "time/dt")
    if q is None or dq is None or dt is None:
        return WARN, ["[WARN] 关节/速度/时间缺失，跳过关节检查"]
    if q.ndim != 2 or q.shape[1] != 6:
        status = max(status, FAIL); msgs.append(f"[FAIL] joints/angles_deg 形状异常: {q.shape}")
        return status, msgs
    if dq.shape != q.shape:
        status = max(status, FAIL); msgs.append(f"[FAIL] joints/vel_deg_s 形状异常: {dq.shape}")
        return status, msgs

    # 速度复算一致性（允许阈值 5 deg/s 的中位绝对误差）
    dtc = np.clip(dt, 1e-3, None)
    re_dq = np.zeros_like(q)
    re_dq[1:] = (q[1:] - q[:-1]) / dtc[1:,None]
    mae = np.nanmedian(np.abs(dq - re_dq))
    if mae > 5.0:
        status = max(status, WARN); msgs.append(f"[WARN] 关节速度与(q差/dt)不一致: median|Δ|={mae:.2f} deg/s (>5)")
    # 角度范围（软限制）：[-3600, 3600] deg
    out = (q < -3600) | (q > 3600)
    n_out = int(np.sum(out))
    if n_out > 0:
        status = max(status, WARN); msgs.append(f"[WARN] 极端关节角 {n_out} 个样本疑似异常 (>|3600| deg)")
    return status, msgs

def check_transforms(fr):
    status = OK; msgs = []
    T_be = safe_get(fr, "poses/T_be_mm")
    T_bc = safe_get(fr, "poses/T_bc_mm")
    if T_be is None or T_bc is None:
        return WARN, ["[WARN] 缺少 T_be/T_bc，跳过 SE(3) 检查"]
    bad_be, bad_bc = 0, 0
    for T in T_be:
        okR, err, det = is_rotation_matrix(T, 5e-3, 5e-2)
        if not okR: bad_be += 1
    for T in T_bc:
        okR, err, det = is_rotation_matrix(T, 5e-3, 5e-2)
        if not okR: bad_bc += 1
    if bad_be or bad_bc:
        status = max(status, WARN)
        msgs.append(f"[WARN] 非正交旋转: T_be {bad_be}/{len(T_be)}, T_bc {bad_bc}/{len(T_bc)}")
    return status, msgs

def check_actions_vels(fr):
    status = OK; msgs = []
    ac_cam = safe_get(fr, "action_cam")
    ac_base= safe_get(fr, "action_base")
    vv_cam = safe_get(fr, "vel_cam")
    vv_base= safe_get(fr, "vel_base")
    dt = safe_get(fr, "time/dt")
    if ac_cam is None or vv_cam is None or dt is None: 
        return WARN, ["[WARN] 缺少 action/vel/dt，跳过动作速度检查"]

    # vel ≈ delta / dt
    dtc = np.clip(dt, 1e-3, None)
    pred_v_cam  = np.zeros_like(vv_cam)
    pred_v_cam[:, :6] = ac_cam[:, :6] / dtc[:, None]
    err_cam = np.nanmedian(np.abs(pred_v_cam[:,:6] - vv_cam[:,:6]), axis=0)

    if np.any(err_cam[:3] > 10.0) or np.any(err_cam[3:6] > 10.0):
        status = max(status, WARN)
        msgs.append(f"[WARN] vel_cam 与 action_cam/dt 不一致: median|Δ|lin={err_cam[:3]}, ang={err_cam[3:6]} (阈值~10)")

    if (ac_base is not None) and (vv_base is not None):
        pred_v_base = np.zeros_like(vv_base)
        pred_v_base[:, :6] = ac_base[:, :6] / dtc[:, None]
        err_base = np.nanmedian(np.abs(pred_v_base[:,:6] - vv_base[:,:6]), axis=0)
        if np.any(err_base[:3] > 10.0) or np.any(err_base[3:6] > 10.0):
            status = max(status, WARN)
            msgs.append(f"[WARN] vel_base 与 action_base/dt 不一致: median|Δ|lin={err_base[:3]}, ang={err_base[3:6]}")

    return status, msgs

def check_targets_geometry(fr, T_ee_cam):
    status = OK; msgs = []
    T_be = safe_get(fr, "poses/T_be_mm")
    t_cam = safe_get(fr, "target/cam_mm")
    t_base= safe_get(fr, "target/base_mm")
    if T_be is None or t_cam is None or t_base is None:
        return WARN, ["[WARN] 缺少目标或位姿，跳过 target 几何一致性"]
    n = len(t_cam); bad = 0; tot = 0
    for i in range(n):
        tc = t_cam[i]   # [3] (mm) or NaN
        tb = t_base[i]
        if np.any(np.isnan(tc)) or np.any(np.isnan(tb)): 
            continue
        p_cam_h = np.r_[tc.astype(float), 1.0]
        pred_b = (T_be[i] @ T_ee_cam @ p_cam_h)[:3]
        e = np.linalg.norm(pred_b - tb.astype(float))
        if e > 30.0:  # 3 cm 阈值
            bad += 1
        tot += 1
    if tot > 0 and bad / tot > 0.1:
        status = max(status, WARN)
        msgs.append(f"[WARN] 目标几何一致性较差: 超过 3cm 的比率 {pct(bad, tot):.1f}% ({bad}/{tot})")
    elif tot == 0:
        msgs.append("[INFO] 目标在大多数帧不可见，未做一致性统计")
    return status, msgs

def check_cmd(fr):
    status = OK; msgs = []
    ctype = safe_get(fr, "cmd/type")
    if ctype is None: 
        return WARN, ["[WARN] 缺少 cmd/*，跳过真实命令检查"]
    # 至少有一部分帧记录了有效命令（类型非0 或 coords/angles 不全 NaN）
    coords = safe_get(fr, "cmd/coords_mmdeg")
    angles = safe_get(fr, "cmd/angles_deg")
    grip   = safe_get(fr, "cmd/gripper")
    valid = 0
    N = len(ctype)
    for i in range(N):
        if ctype[i] != 0: 
            valid += 1; continue
        cc = coords[i] if coords is not None else np.full(6, np.nan)
        aa = angles[i] if angles is not None else np.full(6, np.nan)
        gg = grip[i]   if grip   is not None else -1
        if (not np.all(np.isnan(cc))) or (not np.all(np.isnan(aa))) or (gg in (0,1)):
            valid += 1
    if valid == 0:
        status = max(status, WARN)
        msgs.append("[WARN] cmd/* 全无有效快照（类型为0且坐标角度全NaN且无夹爪状态）")
    return status, msgs

def summarize_nan_outliers(fr):
    status = OK; msgs = []
    keys = [
        "state","action_cam","action_base","vel_cam","vel_base",
        "joints/angles_deg","joints/vel_deg_s",
        "poses/T_be_mm","poses/T_bc_mm",
        "target/cam_mm","target/base_mm"
    ]
    for k in keys:
        if k not in fr: 
            continue
        arr = fr[k][...]
        a = np.asarray(arr, dtype=float)
        total = a.size
        n_nan = int(np.isnan(a).sum())
        if n_nan > 0:
            rate = pct(n_nan, total)
            lvl = WARN if rate < 5.0 else FAIL
            status = max(status, lvl)
            msgs.append(f"[{'WARN' if lvl==WARN else 'FAIL'}] {k} 含 NaN: {rate:.2f}%")
    return status, msgs

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
            s, m = check_targets_geometry(fr, T_ee);   st=max(st,s); report+=m
            s, m = check_cmd(fr);                      st=max(st,s); report+=m
            s, m = summarize_nan_outliers(fr);         st=max(st,s); report+=m

            # 友好总结
            N = fr["state"].shape[0] if "state" in fr else 0
            report.append(f"[INFO] 帧数: {N}")
            if "images/rgb" in fr:
                report.append(f"[INFO] 图像尺寸: {fr['images/rgb'].shape[1:4]}")
            succ = f.attrs.get("success", None); coll = f.attrs.get("collision", None)
            report.append(f"[INFO] success={succ}  collision={coll}")

    except KeyError as e:
        st = FAIL
        report.append(f"[FAIL] 结构缺失: {e}")
    except Exception as e:
        st = FAIL
        report.append(f"[FAIL] 读取异常: {e}")

    return st, "\n".join(report)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="hdf5 文件或目录")
    ap.add_argument("--glob", default="*.hdf5", help="当传目录时的通配符 (默认 *.hdf5)")
    args = ap.parse_args()

    # 收集待检查文件
    files = []
    for p in args.paths:
        if os.path.isdir(p):
            files.extend(glob.glob(os.path.join(p, args.glob)))
        else:
            files.append(p)
    files = sorted(set(files))

    if not files:
        print("请提供待检查的文件或目录，例如：\n  python check_hdf5.py /path/to/dir --glob \"*.hdf5\"")
        return 2

    exit_code = OK
    for f in files:
        st, rep = check_file(f)
        print(rep)
        print()
        exit_code = max(exit_code, st)

    if exit_code == OK:
        print("✅ 全部通过")
    elif exit_code == WARN:
        print("⚠️ 存在警告，请关注上面的提示")
    else:
        print("❌ 存在致命错误，请修复后重试")
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
