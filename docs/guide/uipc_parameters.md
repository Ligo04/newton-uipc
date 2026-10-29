<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# UIPC Solver 参数使用汇总

本文汇总当前 Newton UIPC backend 中 `SolverUIPC`、刚体、软体和布料实际写入或影响 UIPC 的参数。内容基于以下实现文件：

- `newton/_src/solvers/uipc/solver_uipc.py`
- `newton/_src/solvers/uipc/rigid_body.py`
- `newton/_src/solvers/uipc/deformable_body.py`
- `newton/_src/solvers/uipc/cloth.py`
- `newton/_src/solvers/uipc/external_forces.py`
- `newton/_src/sim/builder.py`

本文列出构造参数、scene 默认值、刚体/软体/布料构建时消耗的模型参数，以及每步读取的 State 外力；通过运行时函数临时设置的控制 API 不单独暴露。

## SolverUIPC 全局参数

| 参数 / API | 默认值 | UIPC 目标 / 作用位置 | 说明 |
| --- | --- | --- | --- |
| `backend` | `"cuda"` | `uipc.Engine(backend_name=...)` | 选择 UIPC engine backend。 |
| `workspace` | `"/tmp/newton_uipc"` | `uipc.Engine(..., workspace=...)` | UIPC 输出目录，也用于 dump 和 profile 报告。 |
| `dt` | `1.0 / 60.0` | `scene_config["dt"]` | UIPC 固定步长 [s]。 |
| `scene_config` | `None` | `uipc.Scene(scene_config)` | 未传入时启用全 GPU FusedPCG CUDA Graph 和 MAS；显式传入的配置会保留已指定的 linear solver、graph mode、FEM preconditioner 和 contact constitution，省略 FEM preconditioner 时默认使用 MAS。构造时仍会覆盖 `dt`、`gravity`、部分 contact/newton 默认值。 |
| `SolverUIPC.ADAPTIVE_KAPPA_MIN` | `9.538500988573645e9 Pa` | 当前 UIPC 0.9.0 brick scene 的 adaptive corridor 下限 | 场景质量、尺度、`d_hat` 或时间步改变后该值可能变化；brick example 用它作为固定接触刚度。 |
| `SolverUIPC.ADAPTIVE_KAPPA_MAX` | `9.538500988573645e11 Pa` | 当前 UIPC 0.9.0 brick scene 的 adaptive corridor 上限 | 与下限成 100 倍关系；其他场景应重新读取或计算自己的 corridor。 |
| `kappa` | `100 * MPa` | `AffineBodyConstitution`、关节 builder | 刚体 AffineBody stiffness 参数 [Pa]。 |
| `default_mass_density` | `1000.0` | 刚体 / 软体 fallback 密度 | 当无法从质量和体积估计密度时使用 [kg/m^3]；无几何 articulation 代理用 `mass / density` 推导刚性体积。 |
| `logger_level` | `ULogger.Warn` | `ULogger.set_level()` | UIPC 日志等级。 |
| `dump_enable` | `False` | solver dump 路径 | 控制是否输出 UIPC surface mesh / 调试数据。 |
| `require_profile` | `False` | UIPC timer / report | 开启后记录 step profile，可由 `save_performance_report()` 导出。 |
| `auto_sync_inertia` | `True` | 初始化后的 ABD inertia 同步 | 初始化后把 UIPC ABD 最终质量、COM、惯量同步回 Newton model。 |
| `cloth_soft_position_strength_ratio` | `100.0` | cloth `SoftPositionConstraint` | 布料软位置约束默认强度比例。 |
| `enable_soft_position_constraint` | `True` | cloth / deformable vertices | 是否给布料和软体顶点添加 dormant `SoftPositionConstraint` 属性。 |

### 默认 scene / contact 配置

| 配置项 | 当前默认值 | 说明 |
| --- | --- | --- |
| `scene_config["linear_system"]["solver"]` | `"fused_pcg"` | 使用融合 PCG 求解器。 |
| `scene_config["linear_system"]["use_cuda_graph"]` | `2` | 尝试把一次 FusedPCG 初始化、预条件、迭代和设备侧收敛判断捕获为全 GPU CUDA Graph。 |
| `scene_config["linear_system"]["fem_preconditioner"]` | `"mas"` | 让 UIPC 在内部为所有非 Empty FEM 几何自动分区并使用 MAS；替代 0.0.25 的 Python `mesh_partition()`。 |
| `scene_config["contact"]["constitution"]` | `"ipc"` | 选择 IPC contact；AL-IPC 会关闭 FusedPCG CUDA Graph。 |
| `scene_config["contact"]["enable"]` | `False` | UIPC contact 默认关闭；需要在 scene config 中打开后才启用。 |
| `scene_config["contact"]["d_hat"]` | `0.001` | IPC 接触安全层距离 [m]。 |
| `scene_config["newton"]["velocity_tol"]` | `0.001` | UIPC Newton 迭代速度容差。 |
| `scene_config["newton"]["transrate_tol"]` | `0.01` | UIPC Newton 迭代平移容差。 |
| `scene_config["gravity"]` | 来自 `model.gravity` | 若 model 有 gravity，则写入 UIPC scene。 |
| 默认 contact pair `friction` | `0.5` | 内置 `env/robot/actor/ground` pair 的摩擦系数。 |
| 默认 contact pair `stiffness` | `-1.0` | 启用 UIPC scene-adaptive kappa；有效刚度由场景质量、尺度、`d_hat` 和时间步计算。 |
| `uipc_brick_stacking` contact `stiffness` | `SolverUIPC.ADAPTIVE_KAPPA_MIN` | 显式使用可接受的最小正值，避免该固定互锁场景注册 adaptive reporter。 |
| 默认 contact pair `ccd` | 按 pair 设置 | `env-robot`、`env-actor`、`ground-robot`、`ground-actor`、`robot-actor`、`actor-actor` 默认开启 CCD；同类静态/机器人 pair 多数关闭。 |

显式配置省略 `linear_system` 或其中的 `fem_preconditioner` 时，同样默认使用 MAS。从 `uipc.Scene.default_config()` 创建的完整配置包含原生默认值 `"diag"`，Newton 会保留它；要启用 MAS，在构造 solver 前补上对应设置：

```python
import uipc
from newton.solvers import SolverUIPC

scene_config = uipc.Scene.default_config()
scene_config["linear_system"]["fem_preconditioner"] = "mas"
solver = SolverUIPC(model, scene_config=scene_config)
```

Mode 2 只捕获一次 FusedPCG 求解，不捕获整帧仿真。Newton 外层迭代、碰撞检测与动态接触集合生成、梯度/Hessian 装配、CCD、CFL、line search、Newton 收敛判断和 scene retrieve 仍由 CPU 编排。CUDA 条件图节点不可用时会退回 Mode 1；捕获失败时会退回 block replay 或普通 kernel launch。Mode 1 通常具有更低的帧耗时，Mode 2 的主要用途是让 CPU 不参与 PCG 内层循环。

## 每步输入外力

| Newton 来源 | UIPC 目标 | 说明 |
| --- | --- | --- |
| `state_in.body_f[:, :3]` | `AffineBodyExternalBodyForce` 的平移和 affine 分量 | 世界坐标力 [N]，作用于 COM；按 Newton body 索引映射到各 instance，包含无几何 articulation proxy。 |
| `state_in.body_f[:, 3:]` | `AffineBodyExternalBodyForce` 的 affine 分量 | 关于 COM 的世界坐标力矩 [N·m]；通过原生 affine transform 转换，保留正确的虚功。 |
| `state_in.particle_f` | `FiniteElementExternalForce` 的 vertex `external_force` | 世界坐标力 [N]；按 cloth / deformable 粒子映射读取，未映射的独立粒子忽略。 |
| 非零外力条目 | `external_force/is_constrained = 1` | 每步刷新；零值或缺失的输入数组停用外力，不复用上一帧载荷。与软约束的共享 `is_constrained` 独立。 |

固定的 body / vertex 不响应外力。`step` 保留独立输入 State 的力数组，并清零输出 State 的力数组；原地 stepping 在消费输入后清零。接触力读取继续使用 `CONTACT_F` observable。

## 刚体 / AffineBody 参数

| Newton 来源 / Solver 参数 | UIPC 目标 / 用法 | 默认 / 计算方式 | 说明 |
| --- | --- | --- | --- |
| `model.body_q` | AffineBody instance transform | 初始化时转成 `mat44` | 刚体初始世界位姿。 |
| `model.body_qd` | UIPC ABD velocity sync | 初始化 / step 同步 | 刚体速度状态同步使用。 |
| `model.body_flags` 中 `BodyFlags.KINEMATIC` | instance `is_fixed = 1` | flag 存在时设置 | Newton kinematic body 在 UIPC 中作为 fixed AffineBody instance。 |
| `model.body_mass` | `mass_density` 或 custom ABD mass matrix | `body_mass / mesh_volume`；体积不可用则 `default_mass_density` | 默认路径用密度让 UIPC 计算 ABD；custom inertia 路径直接写 mass matrix。 |
| `model.body_com` | custom ABD mass matrix | 仅 custom inertia 路径使用 | COM 使用 Newton body-local COM。 |
| `model.body_inertia` | custom ABD mass matrix | 仅 custom inertia 路径使用 | 写入前会对惯量矩阵做对称化。 |
| `kappa` | `AffineBodyConstitution.apply_to(...)` | `100 * MPa` | 刚体 AffineBody stiffness [Pa]。 |
| USD `uipc:constitution` / `model.uipc.rigid_model` | 刚体本构选择 | `"abd"` | `NewtonUIPCRigidBodyAPI` 声明该 token；需在 `builder.add_usd(...)` 前注册 UIPC 属性。动态、运动学和带关节的刚体使用 ABD，指定 Empty 会报错。 |
| Collider 上的 USD `uipc:constitution` / `model.uipc.shape_constitution` | 静态碰撞几何的本构选择 | `""`，按归属使用默认值 | `shape_body == -1` 时默认固定 Empty，也可指定固定 ABD；不同本构分开构建。属于刚体的 collider 只能指定 ABD。Plane 使用隐式 halfplane，不接受显式本构选择。 |
| `default_mass_density` | fallback `mass_density` | `1000.0 kg/m^3` | body mass 缺失或 mesh volume 不可用时使用；动态无几何代理以 `mass / density` 作为 ABD 刚性势能的体积系数，保留原始质量、质心和惯量。 |
| `model.shape_body` | body -> collision shape 映射 | 只选属于当前 body 的 shape | world 静态 shape 使用 `shape_body == -1` 路径。 |
| `model.shape_type` | 生成 UIPC mesh / halfplane | 支持 mesh、convex mesh、box、sphere、capsule、cylinder、cone；plane 单独处理 | plane 不作为 AffineBody mesh；ground plane 走 `halfplane`。 |
| `model.shape_transform` | shape 局部变换 | 合成 body mesh | 用于把 shape 几何放到 body-local/world 位置。 |
| `model.shape_scale` | shape 尺寸 / scale | 合成 body mesh | primitive 和 mesh 构造时使用。 |
| `model.shape_source` | mesh / convex mesh 源 | mesh shape 使用 | primitive shape 不依赖该字段。 |
| `model.shape_flags` | 是否参与 UIPC mesh | 只使用 `ShapeFlags.COLLIDE_SHAPES` | 非碰撞 shape 不贡献 UIPC 接触几何。 |
| articulation membership | contact element `robot` | 非 free joint 的 articulation body | 机器人 link 默认归到 `robo_elem`。 |
| free joint membership | contact element `actor` | `JointType.FREE` child body | free-joint body 默认归到 `actor_elem`。 |
| 其他 body | contact element `env` | fallback | 非 articulation / free body 默认归到环境元素。 |

## 软体 / DeformableBody 参数

| Newton 来源 / Solver 参数 | UIPC 目标 / 用法 | 默认 / 计算方式 | 说明 |
| --- | --- | --- | --- |
| `model.particle_q` | UIPC tetmesh vertices | 直接拷贝选中粒子坐标 | 软体几何顶点位置 [m]。 |
| `model.tet_indices` | UIPC tetmesh tetrahedra | 按 `model.uipc.deformable_body_*` rows 或 particle range 选取并重映射 | 软体四面体拓扑。 |
| `model.uipc.deformable_body_{label,world,particle_*,tetrahedron_*,triangle_*,edge_*,density}` | 分组构建 deformable object | `SolverUIPC.register_custom_attributes(builder)` 在 `finalize()` 时生成 | 使用 inclusive `first/last` 索引保存每个 authored soft body，避免不同软体混到一个 UIPC geometry。 |
| `model.particle_mass` | mass density 估计；fixed marker | `sum(mass) / tet_volume`；失败时 `default_mass_density` | `particle_mass <= 0.0` 的粒子会写成 UIPC vertex `is_fixed = 1`。 |
| `default_mass_density` | 所选本构 `apply_to(..., mass_density=...)` 的密度 fallback | `1000.0 kg/m^3` | 质量或体积不可用时使用。 |
| `model.uipc.deformable_model[i]` | 第 `i` 个 `uipc:deformable_body` row 的软体四面体本构选择 | `"stable_neo_hookean"` | 需在 `finalize()` 前调用 `SolverUIPC.register_custom_attributes(builder)`；支持 `"stable_neo_hookean"`、`"arap"` 和 `"empty"`。 |
| USD `TetMesh` 上的 `uipc:constitution` | 导入到对应的 `model.uipc.deformable_model[i]` | `"stable_neo_hookean"` | 由实验性 `NewtonUIPCDeformableAPI` 声明；也支持同名自定义 token。需在 `builder.add_usd(...)` 前注册 UIPC 属性；仅用于四面体软体。 |
| `model.tet_materials[:, 0]` (`k_mu`) | UIPC tet `mu` | 写入 `(4 / 3) * k_mu` | UIPC StableNeoHookean 的 `mu` 不是原样写入，而是做了变换。 |
| `model.tet_materials[:, 1]` (`k_lambda`) | UIPC tet `lambda` | 写入 `k_lambda + (5 / 6) * k_mu` | UIPC StableNeoHookean 的 `lambda` 会叠加部分 `k_mu`。 |
| `model.tet_materials[:, 0]` (`k_mu`) 且对应 `model.uipc.deformable_model[i] == "arap"` | UIPC tet `kappa` | 原样写入 `k_mu` | ARAP 不使用 StableNeoHookean 的 `mu/lambda` 变换。 |
| `model.tet_materials` 且对应 `model.uipc.deformable_model[i] == "empty"` | UIPC `Empty`（UID 0） | 不写入弹性参数 | 保留几何和质量，不添加弹性能量；仍可配置接触和软位置约束。 |
| `model.tet_materials[:, 2]` (`k_damp`) | 当前 UIPC builder 未写入 | `ModelBuilder.default_tet_k_damp = 0.0` | Newton 会保存该列，但当前 UIPC deformable builder 没有映射 damping 参数。 |
| `enable_soft_position_constraint` | `SoftPositionConstraint.apply_to(sc)` | `True` | 为软体顶点添加软位置约束属性；具体目标值由内部同步逻辑写入。 |
| contact element | `actor_elem` | 默认 | 当前 solver 构建 soft body 时默认归入 actor contact element。 |

## 布料 / Cloth 参数

| Newton 来源 / Solver 参数 | UIPC 目标 / 用法 | 默认 / 计算方式 | 说明 |
| --- | --- | --- | --- |
| `model.particle_q` | UIPC trimesh vertices | 直接拷贝选中 cloth 粒子坐标 | 布料几何顶点位置 [m]。 |
| `model.tri_indices` | UIPC trimesh faces | 按 `model.uipc.cloth_*` rows 或 legacy heuristic 选取并重映射 | 布料三角面拓扑。闭合三角网格会报错，建议闭合体使用软体或刚体。 |
| `model.uipc.cloth_{label,world,particle_*,triangle_*,edge_*,spring_*,surface_density}` | 分组构建 cloth object | `SolverUIPC.register_custom_attributes(builder)` 在 `finalize()` 时生成 | 使用 inclusive `first/last` 索引保存每个 authored cloth，避免多个 cloth 混成一个 UIPC geometry。 |
| `model.uipc.cloth_model[i]` | 第 `i` 个 `uipc:cloth` row 的 membrane constitution | 默认 `StrainLimitingBaraffWitkinShell`；可选 `NeoHookeanShell` | 需在 `finalize()` 前调用 `SolverUIPC.register_custom_attributes(builder)`；两种 membrane 后续都会写入 `mu/lambda` 三角属性。 |
| 布料 simulation `Mesh` 上的 USD `uipc:constitution` | 导入到对应的 `model.uipc.cloth_model[i]` | `"strain_limiting_baraff_witkin"`；可选 `"neo_hookean"` | 由实验性 `NewtonUIPCClothAPI` 声明，也支持同名自定义 token；需在 `builder.add_usd(...)` 前注册 UIPC 属性。只选择膜本构，仍单独添加 `DiscreteShellBending`。 |
| `model.particle_radius` | vertex `thickness` 和 `volume` 修正 | 直接读取 cloth 粒子的 `particle_radius`；负值报错；缺失时回退默认厚度 | 布料 thickness 与粒子碰撞半径保持一致。 |
| `model.particle_mass` + `model.tri_areas` | membrane `mass_density` | `sum(particle_mass) / sum(tri_area) / thickness`；失败时 `100.0` | UIPC shell 使用体密度 [kg/m^3]。 |
| `model.tri_materials[:, 0]` (`tri_ke`) | UIPC triangle `mu` | 当前实现原样写入 | 虽然注释提到 Young's modulus，但实际代码把该列写到 `mu`。 |
| `model.tri_materials[:, 1]` (`tri_ka`) | UIPC triangle `lambda` | 当前实现原样写入 | 当前实现把 area stiffness / 第二列写到 `lambda`。 |
| `model.tri_materials[:, 2]` (`tri_kd`) | 当前 UIPC builder 未写入 | `ModelBuilder.default_tri_kd = 10.0` | Newton 会保存 damping 列，但当前 UIPC cloth builder 没有映射。 |
| `model.tri_materials[:, 3]` (`tri_drag`) | 当前 UIPC builder 未写入 | `ModelBuilder.default_tri_drag = 0.0` | aerodynamic drag 当前未映射到 UIPC cloth。 |
| `model.tri_materials[:, 4]` (`tri_lift`) | 当前 UIPC builder 未写入 | `ModelBuilder.default_tri_lift = 0.0` | aerodynamic lift 当前未映射到 UIPC cloth。 |
| `DiscreteShellBending.apply_to(...)` | edge `bending_stiffness` 初值 | builder 默认 `0.01` | 所有 UIPC bending edge 先应用默认 bending stiffness。 |
| `model.edge_bending_properties[:, 0]` (`edge_ke`) | UIPC edge `bending_stiffness` | `ModelBuilder.default_edge_ke = 100.0`，或 cloth/edge API 显式传入 | `_write_edge_bending_stiffness()` 用这一列覆盖 UIPC 每条 bending edge stiffness。 |
| `model.edge_bending_properties[:, 1]` (`edge_kd`) | 当前 UIPC builder 未写入 | `ModelBuilder.default_edge_kd = 0.0` | Newton 会保存 damping，但当前 UIPC cloth builder 只使用 `edge_ke`。 |
| `enable_soft_position_constraint` | `SoftPositionConstraint.apply_to(sc, strength_ratio)` | `True` | 为布料顶点添加软位置约束属性；具体目标值由内部同步逻辑写入。 |
| `cloth_soft_position_strength_ratio` | cloth `SoftPositionConstraint` 默认强度 | `100.0` | 只在布料 builder 中作为默认 ratio 传入。 |
| contact element | `actor_elem` | 默认 | 当前 solver 构建 cloth 时默认归入 actor contact element。 |

## ROD 中心线参数映射

`SolverUIPC` 将 capsule 段和 ROD 关节转换为 `HookeanSpring`＋可选
`KirchhoffRodBending`。段不再重复创建 ABD；求解后由中心线重建段姿态、
COM 速度和关节状态。Newton capsule 保留静止长度，FEM 中心线允许拉伸，
两者的几何并非完全一致。初始化会提示这项近似。

| Newton 输入 | UIPC 对应项 | 换算与限制 |
| --- | --- | --- |
| simulation `BasisCurves` 的 `uipc:constitution` | `model.uipc.rod_model[i]`，内置 Curve frequency | 默认且仅支持 `"hookean_spring"`；由 `NewtonUIPCRodAPI` 声明。导入前注册 UIPC 属性，曲线需已有 `PhysicsCurvesDeformableSimAPI`。 |
| ROD 轴向 `joint_target_ke` | edge `kappa` | `EA = k_stretch * L_dual`，再除以 `pi*r^2`；每条边取相邻关节恢复的 `EA` 平均值。孤立段使用 `EA = 1e5 * rest_length`。 |
| ROD 两个弯曲轴的 `joint_target_ke` | vertex `bending_stiffness` | 两轴必须相等；`EI = k_bend * L_dual`，再除以 `pi*r^4/4`。全部为零时不添加弯曲本构。 |
| `body_mass` | vertex `mass` | 每段质量各分一半到两端，共享节点累加。 |
| `body_qd`、`State.body_f` | 节点速度、外力 | 将 COM 速度和角速度换成端点速度；合力及横向力矩分配到端点。轴向力矩没有响应。 |
| 运动学／零质量段、世界 FIXED／端点 BALL 根关节 | 固定节点 | 固定段约束两个端点，BALL 根约束一个端点；运动学目标逐步更新。 |
| 剪切、扭转、材料阻尼、静止曲率 | 不映射 | 初始化明确警告；静止曲率为零。姿态按中心线平行输运，读回角速度不含轴向自旋。 |

`L_dual` 是相邻段的平均静止长度。支持非分叉的开放链和闭环；要求每段
恰好一个 capsule、统一正半径和初始重合的关节端点。分叉、回折弯曲节点、禁用的 ROD
关节、与其他刚体的连接、逐段 contact-element 覆盖和杆 proxy 会报错。
可使用原生接触；含杆场景暂不提供 `CONTACT_F` 观测。`reset()` 支持按
world mask 分别更新 body 位置和速度。

## ModelBuilder 中与 UIPC 相关的材料默认值

| Builder 默认值 | 当前值 | 主要影响 | UIPC 当前是否使用 |
| --- | --- | --- | --- |
| `default_tri_ke` | `100.0` | `tri_materials[:, 0]` | 使用：写入 cloth triangle `mu`。 |
| `default_tri_ka` | `100.0` | `tri_materials[:, 1]` | 使用：写入 cloth triangle `lambda`。 |
| `default_tri_kd` | `10.0` | `tri_materials[:, 2]` | 未使用。 |
| `default_tri_drag` | `0.0` | `tri_materials[:, 3]` | 未使用。 |
| `default_tri_lift` | `0.0` | `tri_materials[:, 4]` | 未使用。 |
| `default_edge_ke` | `100.0` | `edge_bending_properties[:, 0]` | 使用：写入 cloth edge `bending_stiffness`。 |
| `default_edge_kd` | `0.0` | `edge_bending_properties[:, 1]` | 未使用。 |
| `default_tet_k_mu` | `1.0e3` | `tet_materials[:, 0]` | 使用：变换后写入 deformable tet `mu`。 |
| `default_tet_k_lambda` | `1.0e3` | `tet_materials[:, 1]` | 使用：变换后写入 deformable tet `lambda`。 |
| `default_tet_k_damp` | `0.0` | `tet_materials[:, 2]` | 未使用。 |
| `default_tet_density` | `1.0` | soft body particle mass authoring | 间接使用：影响 `particle_mass`，进而影响 UIPC mass density 估计。 |

## 注意事项

- `_write_edge_bending_stiffness()` 使用的是 `model.edge_bending_properties[edge_id, 0]`，也就是 `edge_ke`；`edge_kd` 当前没有写入 UIPC。
- 布料的 `tri_ke/tri_ka` 当前实现是直接写入 UIPC triangle `mu/lambda`，不是在 UIPC builder 中再转换为 Young's modulus / Poisson ratio。
- 软体默认使用 StableNeoHookean；此时 `tet_materials` 会做本构所需的参数变换：`mu = (4/3) * k_mu`，`lambda = k_lambda + (5/6) * k_mu`。选择 ARAP 时，`k_mu` 写入 UIPC tet `kappa`；选择 Empty 时忽略 `tet_materials`，不添加弹性能量。
- `SoftPositionConstraint` 只有在 `enable_soft_position_constraint=True` 时才会添加属性；目标位置、启用标记和强度比例属于运行时约束数据，不在本表展开为公开配置项。
- UIPC cloth 的 shell thickness 现在直接来源于 `model.particle_radius`，因此 cloth 粒子半径和 UIPC 厚度是同一参数。
- UIPC 全局 contact 默认关闭；即使 contact tabular 已经配置 pair，也只有 `scene_config["contact"]["enable"]` 打开后才会启用接触。
