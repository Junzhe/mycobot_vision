import argparse, sys, os, glob
import numpy as np
import h5py

OK, WARN, FAIL = 0, 1, 2

def pct(x, n): 
    return 0.0 if n == 0 else 100.0 * x / n

def is_rotation_matrix(Rm, tol_orth=1e-3, tol_det=1e-2):
    R = Rm[:3,:3]
    err = np.linalg.norm(R.T @ R - np.eye(3), ord='fro')
    d = np.linalg.det(R)
    return (err < tol_orth) and (abs(d-1.0) < tol_det), err, d

def safe_get(g, name, default=None):
    try:
        if name in g: return g[name][...]
    except Exception:
        pass
    return default

# ---------------- 基础检查 ----------------

def check_attrs(f):
    status, msgs = OK, []
    need = ["st_pos_unit","st_rot_unit","ac_pos_unit","ac_rot_unit"]
    for k in need:
        if k not in f.attrs:
            status = max(status, WARN)
            msgs.append(f"[WARN] file.attrs 缺少 {k}")
    up = f.attrs.get("st_pos_unit", "")
    ur = f.attrs.get("st_rot_unit", "")
    if isinstance(up, bytes): up = up.decode()
    if isinstance(ur, bytes): ur = ur.decode()
    if up and up not in ("mm",):
        status = max(status, WARN); msgs.append(f"[WARN] st_pos_unit={up} (期望 mm)")
    if ur and ur not in ("deg",):
        status = max(status, WARN); msgs.append(f"[WARN] st_rot_unit={ur} (期望 deg)")

    T_ee_cam = f.attrs.get("T_ee_cam_mm", None)
    if T_ee_cam is None:
        msgs.append("[WARN] 未找到 attrs['T_ee_cam_mm']，几何一致性只能近似检查")
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
    need = [
        "images/rgb","images/mask",
        "state",
        "action_cam","action_base",
        "vel_cam","vel_base",
        "time/ts","time/dt",
        "joints/angles_deg","joints/vel_deg_s",
        "poses/T_be_mm","poses/T_bc_mm",
        "cmd/type","cmd/speed","cmd/coords_mmdeg","cmd/angles_deg","cmd/gripper","cmd/time",
        "cond/target","cond/phase",
        "target/cam_mm","target/base_mm","target/visible"
    ]
    missing = [k for k in need if k not in fr]
    if missing:
        status = max(status, FAIL)
        msgs.append(f"[FAIL] 缺少数据集: {missing}")

    def shape(n): return fr[n].shape if n in fr else None
    if "images/rgb" in fr and "images/mask" in fr:
        s_rgb, s_mask = shape("images/rgb"), shape("images/mask")
        if not (s_rgb and len(s_rgb)==4 and s_rgb[-1]==3):
            status = max(status, FAIL); msgs.append(f"[FAIL] images/rgb 形状异常: {s_rgb} (应 [N,H,W,3])")
        if not (s_mask and len(s_mask)==3):
            status = max(status, FAIL); msgs.append(f"[FAIL] images/mask 形状异常: {s_mask} (应 [N,H,W])")
        if s_rgb and s_mask and s_rgb[:3]!=s_mask[:3]:
            status = max(status, FAIL); msgs.append(f"[FAIL] 图像与掩码维度不一致: rgb={s_rgb}, mask={s_mask}")

    main = [k for k in need if k in fr]
    lens = [fr[k].shape[0] for k in main if fr[k].ndim>=1]
    if lens:
        uniq = set(lens)
        if len(uniq)!=1:
            status = max(status, FAIL)
            msgs.append(f"[FAIL] 主序列长度不一致: " + ", ".join(f"{k}:{fr[k].shape}" for k in main))
    else:
        status = max(status, FAIL); msgs.append("[FAIL] 未找到任何主序列数据集")
    return status, msgs

def check_time(fr):
    status, msgs = OK, []
    ts, dt = safe_get(fr,"time/ts"), safe_get(fr,"time/dt")
    if ts is None or dt is None: return FAIL, ["[FAIL] 缺少 time/ts 或 time/dt"]
    if ts.ndim!=1 or dt.ndim!=1 or len(ts)!=len(dt):
        return FAIL, [f"[FAIL] ts/dt 形状异常: ts={ts.shape}, dt={dt.shape}"]
    diff = np.diff(ts)
    n_nonmono = int(np.sum(diff<=0))
    if n_nonmono>0:
        status=max(status,FAIL); msgs.append(f"[FAIL] ts 非严格递增: {n_nonmono}/{len(diff)}")
    dt_from_ts = np.r_[diff, diff[-1] if len(diff) else 0]
    bad = np.abs(dt_from_ts - dt) > 0.003
    if int(np.sum(bad))>0:
        status=max(status,WARN); msgs.append(f"[WARN] dt 与 ts 差分不一致: {int(np.sum(bad))}/{len(dt)} 处 >3ms")
    out = (dt<0.002)|(dt>0.2)
    if int(np.sum(out))>0:
        status=max(status,WARN); msgs.append(f"[WARN] dt 超常: {int(np.sum(out))}/{len(dt)} ∉ [2ms,200ms]")
    return status, msgs

def check_images(fr):
    status, msgs = OK, []
    rgb, msk = safe_get(fr,"images/rgb"), safe_get(fr,"images/mask")
    if rgb is None or msk is None: return WARN, ["[WARN] 无图像/掩码，跳过图像检查"]
    N,H,W,C = rgb.shape
    if C!=3: status=max(status,FAIL); msgs.append(f"[FAIL] RGB 通道数={C} (应3)")
    samp = rgb.reshape(-1,3)[::max(1,(N*H*W)//20000)]
    if (samp.min()<0) or (samp.max()>255):
        status=max(status,FAIL); msgs.append(f"[FAIL] RGB 值域异常: [{samp.min()},{samp.max()}]")
    mu = np.unique(msk.reshape(-1)[::max(1,(N*H*W)//50000)])
    bad = [v for v in mu if v not in (0,255)]
    if bad: status=max(status,WARN); msgs.append(f"[WARN] 掩码存在非 0/255 值: {bad[:10]}")
    return status, msgs

def check_joints(fr):
    status, msgs = OK, []
    q, dq, dt = safe_get(fr,"joints/angles_deg"), safe_get(fr,"joints/vel_deg_s"), safe_get(fr,"time/dt")
    if q is None or dq is None or dt is None: return WARN, ["[WARN] 关节/速度/时间缺失，跳过关节检查"]
    if q.ndim!=2 or q.shape[1]!=6:
        status=max(status,FAIL); msgs.append(f"[FAIL] joints/angles_deg 形状异常: {q.shape}"); return status,msgs
    if dq.shape!=q.shape:
        status=max(status,FAIL); msgs.append(f"[FAIL] joints/vel_deg_s 形状异常: {dq.shape}"); return status,msgs
    dtc = np.clip(dt,1e-3,None)
    re_dq = np.zeros_like(q); re_dq[1:] = (q[1:]-q[:-1])/dtc[1:,None]
    mae = np.nanmedian(np.abs(dq - re_dq))
    if mae>5.0: status=max(status,WARN); msgs.append(f"[WARN] 关节速度与差分不一致: median|Δ|={mae:.2f} deg/s")
    out = (q<-3600)|(q>3600)
    if int(np.sum(out))>0: status=max(status,WARN); msgs.append(f"[WARN] 关节角极端值: {int(np.sum(out))} 个点")
    return status, msgs

def check_transforms(fr):
    status, msgs = OK, []
    T_be, T_bc = safe_get(fr,"poses/T_be_mm"), safe_get(fr,"poses/T_bc_mm")
    if T_be is None or T_bc is None: return WARN, ["[WARN] 缺少 T_be/T_bc，跳过 SE(3) 检查"]
    bad_be = sum(1 for T in T_be if not is_rotation_matrix(T,5e-3,5e-2)[0])
    bad_bc = sum(1 for T in T_bc if not is_rotation_matrix(T,5e-3,5e-2)[0])
    if bad_be or bad_bc:
        status=max(status,WARN); msgs.append(f"[WARN] 非正交旋转: T_be {bad_be}/{len(T_be)}, T_bc {bad_bc}/{len(T_bc)}")
    return status, msgs

def check_actions_vels(fr):
    status, msgs = OK, []
    ac_cam, ac_base = safe_get(fr,"action_cam"), safe_get(fr,"action_base")
    vv_cam, vv_base = safe_get(fr,"vel_cam"), safe_get(fr,"vel_base")
    dt = safe_get(fr,"time/dt")
    if ac_cam is None or vv_cam is None or dt is None:
        return WARN, ["[WARN] 缺少 action/vel/dt，跳过动作速度检查"]
    dtc = np.clip(dt,1e-3,None)
    pred_v_cam = np.zeros_like(vv_cam); pred_v_cam[:,:6] = ac_cam[:,:6]/dtc[:,None]
    err_cam = np.nanmedian(np.abs(pred_v_cam[:,:6]-vv_cam[:,:6]),axis=0)
    if np.any(err_cam[:3]>10.0) or np.any(err_cam[3:6]>10.0):
        status=max(status,WARN); msgs.append(f"[WARN] vel_cam 与 action_cam/dt 不一致: lin={err_cam[:3]}, ang={err_cam[3:6]}")

    # 动作幅度/速度上限（经验阈值，训练前可用于清洗）
    if np.nanmax(np.abs(ac_base[:,:3])) > 120:  # >120 mm/步
        status=max(status,WARN); msgs.append("[WARN] action_base 平移步幅>120mm，疑似跳变")
    if np.nanmax(np.abs(vv_base[:,:3])) > 600:  # >600 mm/s
        status=max(status,WARN); msgs.append("[WARN] vel_base 线速度>600mm/s，疑似异常")

    if (ac_base is not None) and (vv_base is not None):
        pred_v_base = np.zeros_like(vv_base); pred_v_base[:,:6] = ac_base[:,:6]/dtc[:,None]
        err_base = np.nanmedian(np.abs(pred_v_base[:,:6]-vv_base[:,:6]),axis=0)
        if np.any(err_base[:3]>10.0) or np.any(err_base[3:6]>10.0):
            status=max(status,WARN); msgs.append(f"[WARN] vel_base 与 action_base/dt 不一致: lin={err_base[:3]}, ang={err_base[3:6]}")
    return status, msgs

# ---------------- 训练可用性增强检查 ----------------

def check_target_visibility_and_phase(fr, T_ee_cam):
    status, msgs = OK, []
    vis = safe_get(fr,"target/visible")
    t_cam = safe_get(fr,"target/cam_mm")
    t_base = safe_get(fr,"target/base_mm")
    T_be = safe_get(fr,"poses/T_be_mm")
    phase = safe_get(fr,"cond/phase")

    if vis is None or t_cam is None or t_base is None or T_be is None or phase is None:
        return WARN, ["[WARN] 缺 target/visible 或 cond/phase，跳过可见性一致性检查"]

    N = len(vis)
    vis_rate = float(np.mean(vis>0)) if N>0 else 0.0
    msgs.append(f"[INFO] target 可见率: {vis_rate*100:.1f}%")

    # 可见时的几何一致性
    ok_idx = np.where((vis>0) & (~np.any(np.isnan(t_cam),axis=1)) & (~np.any(np.isnan(t_base),axis=1)))[0]
    bad = 0
    for i in ok_idx:
        pred_b = (T_be[i] @ T_ee_cam @ np.r_[t_cam[i].astype(float),1.0])[:3]
        e = np.linalg.norm(pred_b - t_base[i].astype(float))
        if e > 30.0: bad += 1
    if len(ok_idx)>0 and bad/len(ok_idx) > 0.1:
        status = max(status, WARN)
        msgs.append(f"[WARN] 可见帧中 target 几何误差>3cm 的比例 {pct(bad,len(ok_idx)):.1f}%")
    elif len(ok_idx)==0:
        msgs.append("[INFO] target 可见帧太少，跳过几何一致性统计")

    # phase 合法性（非递减，且只取 0/1/2）
    uniq = set(int(x) for x in np.unique(phase))
    if not uniq.issubset({0,1,2}):
        status=max(status,WARN); msgs.append(f"[WARN] cond/phase 出现非 {0,1,2} 值: {sorted(list(uniq))}")
    if np.any(np.diff(phase) < -0.5):  # 允许浮点
        status=max(status,WARN); msgs.append("[WARN] cond/phase 非单调（阶段回退）")

    # 长时间不可见且未 confirmed 的区间
    long_gap = 0
    max_gap = 0
    for i in range(N):
        if (vis[i]==0) and (phase[i]<2): 
            long_gap += 1; max_gap = max(max_gap, long_gap)
        else:
            long_gap = 0
    if max_gap >= max(10, int(0.5*N/10)):
        status=max(status,WARN); msgs.append(f"[WARN] 抓取前存在较长不可见区段（长度~{max_gap} 帧）")

    return status, msgs

def check_cmd_coverage(fr):
    status, msgs = OK, []
    ctype = safe_get(fr,"cmd/type")
    coords = safe_get(fr,"cmd/coords_mmdeg")
    angles = safe_get(fr,"cmd/angles_deg")
    grip   = safe_get(fr,"cmd/gripper")
    speed  = safe_get(fr,"cmd/speed")
    if ctype is None: 
        return WARN, ["[WARN] 缺少 cmd/*，无法校验真实命令覆盖"]

    N = len(ctype)
    has_any = 0
    kinds = set()
    for i in range(N):
        t = int(ctype[i])
        if t!=0: kinds.add(t); has_any += 1
        else:
            if coords is not None and not np.all(np.isnan(coords[i])): has_any += 1
            if angles is not None and not np.all(np.isnan(angles[i])): has_any += 1
            if grip is not None and (grip[i] in (0,1)): has_any += 1
    cover = has_any / max(1,N)
    if cover < 0.2:
        status=max(status,WARN); msgs.append(f"[WARN] cmd/* 覆盖率偏低: {cover*100:.1f}%")
    else:
        msgs.append(f"[INFO] cmd 覆盖率: {cover*100:.1f}%  种类: {sorted(list(kinds))}  平均速度: {np.nanmean(speed) if speed is not None else 'NA'}")
    return status, msgs

def check_gripper_phase_success(fr, f_attrs):
    status, msgs = OK, []
    grip = safe_get(fr,"cmd/gripper")
    phase = safe_get(fr,"cond/phase")
    if grip is None or phase is None:
        return WARN, ["[WARN] 缺少 cmd/gripper 或 cond/phase，跳过夹爪-阶段一致性"]

    # 找“闭合”时刻
    close_idx = np.where(grip==1)[0]
    if close_idx.size>0:
        t_close = int(close_idx[0])
        if np.any(phase[max(0,t_close-5):t_close+5] >= 2):
            msgs.append("[INFO] 夹爪闭合附近 phase≈confirmed ✓")
        else:
            status=max(status,WARN); msgs.append("[WARN] 夹爪闭合附近未见 phase=confirmed")
    else:
        msgs.append("[INFO] 未检测到夹爪闭合命令")

    succ = f_attrs.get("success", None)
    coll = f_attrs.get("collision", None)
    msgs.append(f"[INFO] success={succ}  collision={coll}")
    return status, msgs

def summarize_nan_outliers(fr):
    status, msgs = OK, []
    keys = [
        "state","action_cam","action_base","vel_cam","vel_base",
        "joints/angles_deg","joints/vel_deg_s",
        "poses/T_be_mm","poses/T_bc_mm",
        "target/cam_mm","target/base_mm","target/visible"
    ]
    for k in keys:
        if k not in fr: 
            continue
        a = np.asarray(fr[k], dtype=float)
        rate = float(np.isnan(a).sum()) / max(1,a.size)
        if rate>0:
            lvl = WARN if rate < 0.05 else FAIL
            status=max(status,lvl)
            msgs.append(f"[{'WARN' if lvl==WARN else 'FAIL'}] {k} 含 NaN: {rate*100:.2f}%")
    return status, msgs

# ---------------- 单文件检查总控 ----------------

def check_file(path):
    st = OK
    report = [f"=== {os.path.basename(path)} ==="]
    try:
        with h5py.File(path, "r") as f:
            if "frames" not in f:
                report.append("[FAIL] 根下无 'frames' 组")
                return FAIL, "\n".join(report)
            fr = f["frames"]

            s, m, T_ee = check_attrs(f);                     st=max(st,s); report+=m
            s, m = check_basic_shapes(fr);                   st=max(st,s); report+=m
            s, m = check_time(fr);                           st=max(st,s); report+=m
            s, m = check_images(fr);                         st=max(st,s); report+=m
            s, m = check_joints(fr);                         st=max(st,s); report+=m
            s, m = check_transforms(fr);                     st=max(st,s); report+=m
            s, m = check_actions_vels(fr);                   st=max(st,s); report+=m
            s, m = check_target_visibility_and_phase(fr,T_ee); st=max(st,s); report+=m
            s, m = check_cmd_coverage(fr);                   st=max(st,s); report+=m
            s, m = check_gripper_phase_success(fr, f.attrs); st=max(st,s); report+=m
            s, m = summarize_nan_outliers(fr);               st=max(st,s); report+=m

            # 友好总结
            N = fr["state"].shape[0] if "state" in fr else 0
            report.append(f"[INFO] 帧数: {N}")
            if "images/rgb" in fr:
                report.append(f"[INFO] 图像尺寸: {fr['images/rgb'].shape[1:4]}")

            # 训练可用性摘要
            vis = safe_get(fr,"target/visible")
            vis_rate = float(np.mean(vis>0)) if vis is not None and len(vis)>0 else 0.0
            dt = safe_get(fr,"time/dt")
            dt_out = int(np.sum((dt<0.002)|(dt>0.2))) if dt is not None else 0
            report.append(f"[SUMMARY] trainable≈ "
                          f"可见率={vis_rate*100:.1f}% | "
                          f"异常dt={dt_out} | "
                          f"无 FAIL 且 WARN 可接受则可直接训练")

    except KeyError as e:
        st = FAIL; report.append(f"[FAIL] 结构缺失: {e}")
    except Exception as e:
        st = FAIL; report.append(f"[FAIL] 读取异常: {e}")
    return st, "\n".join(report)

# ---------------- 主函数 ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="hdf5 文件或目录")
    ap.add_argument("--glob", default="*.hdf5", help="当传目录时的通配符 (默认 *.hdf5)")
    args = ap.parse_args()

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
        print(rep); print()
        exit_code = max(exit_code, st)

    if exit_code == OK:
        print("✅ 全部通过")
    elif exit_code == WARN:
        print("⚠️ 存在警告，但通常可通过 delta/dt 或重采样在训练时对冲")
    else:
        print("❌ 存在致命错误，请修复后重试")
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
