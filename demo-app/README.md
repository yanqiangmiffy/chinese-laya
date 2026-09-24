# 中文 Laya 场景验证台

React + FastAPI 的本地推理 Demo。网页把中文 state 和 typed questions 发给本机 FastAPI；后端直接用仓库中的 Laya 源码加载训练输出目录。模型文件不会打进镜像，也不会发到外部服务。

页面沿用多场景评测台的 9 个标签与 27 条样例。可以逐条运行、跑单场景或跑全部样例；返回答案、概率、置信度和参考标签对照会显示在同一页。参考标签只留在前端，不会发送给模型；自定义输入可以推理，但不计入参考标签一致率。

## 本机启动（Windows Conda）

在两个 PowerShell 窗口里，从仓库根目录分别运行：

```powershell
.\demo-app\start-backend.ps1
```

```powershell
.\demo-app\start-frontend.ps1
```

等待后端日志出现 `Laya ready`，浏览器打开 <http://127.0.0.1:5173>。

## 本机启动（Linux / Ubuntu Conda）

在两个终端里分别执行：

```bash
./demo-app/start-backend.sh
```

```bash
./demo-app/start-frontend.sh
```

浏览器打开 <http://127.0.0.1:5173>。API 文档在 <http://127.0.0.1:18000/docs>，健康检查在 <http://127.0.0.1:18000/api/health>。

启动脚本使用已有 `base`，默认 `DEVICE=cuda`、`workers=1`。启动器会让 Conda base 的包优先于用户级包，避免同名包版本冲突，同时保留用户目录中缺失小依赖作为后备；不创建新环境，也不改写 Conda 的整套依赖。默认 checkpoint 是仓库中已存在的 `runs/laya-typed-decisions-zh`，只读加载，不会写入权重。

如果你要运行另一个训练输出目录（例如日志中提到的 `laya-typed-decisions-zh-v2`），PowerShell 中先设置：

```powershell
$env:MODEL_PATH = 'D:\models\runs\laya-typed-decisions-zh-v2'
.\demo-app\start-backend.ps1
```

Linux 中可用绝对路径：

```bash
MODEL_PATH=/data/users/yanqiang/chinese-laya/runs/laya-typed-decisions-zh-v2 ./demo-app/start-backend.sh
```

页面阈值只在前端生成“建议复核”提示，不会触发付款、发送、删除或其他外部动作。`noul` 的 P(true)、`answer_confidence` 与 `confidence` 分开展示；标签一致率和阈值都应在独立中文验证集上确认。
