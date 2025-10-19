# AIME Training - 新服务器部署总结

**服务器信息:** 2.tcp.ngrok.io:15590  
**密码:** Zbr31W1UD9o8  
**部署时间:** 2025-10-16  
**状态:** ✅ **95%完成，Qwen模型下载中**

---

## ✅ 已完成的部署

### 1. 目录结构
```
/root/
├── AFlow/              # AFlow代码库
│   └── data/
│       └── AIME_2024.jsonl  # 30个AIME问题
├── integration/        # 集成代码
│   ├── deep_train_real_workflow.py
│   ├── deep_workflow_env.py
│   ├── workflow_parser.py
│   ├── workflow_evaluator.py
│   ├── aime_config.yaml
│   ├── aime_evaluator.py
│   └── ...
└── models/             # 模型目录
    └── Qwen2.5-7B-Instruct/  # ⏳ 下载中 (15GB)
```

### 2. 已上传文件
✅ deep_train_real_workflow.py  
✅ deep_workflow_env.py  
✅ workflow_parser.py  
✅ workflow_evaluator.py  
✅ workflow_prompt_manager.py  
✅ unified_state.py  
✅ aime_config.yaml  
✅ aime_evaluator.py  
✅ aime_prompt_manager.py  
✅ test_aime_workflow.py  

### 3. AFlow代码库
✅ 已下载完成

### 4. AIME数据集
✅ 已下载完成（30个问题，49KB）

### 5. Python依赖
✅ 已安装: torch, transformers, datasets, pyyaml, numpy, asyncio

---

## ⏳ 正在进行

### Qwen模型下载
- **状态:** 后台下载中
- **大小:** 15GB
- **预计时间:** 15-30分钟
- **日志文件:** /root/qwen_download.log

**监控下载进度:**
```bash
ssh root@2.tcp.ngrok.io -p 15590
tail -f /root/qwen_download.log
```

---

## 🚀 启动训练（下载完成后）

### 步骤1: 连接服务器
```bash
ssh root@2.tcp.ngrok.io -p 15590
# 密码: Zbr31W1UD9o8
```

### 步骤2: 检查Qwen模型下载状态
```bash
ls -lh /root/models/Qwen2.5-7B-Instruct/ | head -10
# 应该看到约15GB的模型文件
```

### 步骤3: 启动AIME训练
```bash
cd /root/integration

# 设置环境变量
export PYTHONPATH=/root/AFlow:/root/integration:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH

# 启动训练（后台运行）
nohup python3 deep_train_real_workflow.py --config aime_config.yaml > aime_training.log 2>&1 &

# 等待几秒
sleep 5

# 查看初始日志
tail -50 aime_training.log
```

### 步骤4: 验证训练正常运行
```bash
# 检查进程
ps aux | grep deep_train_real_workflow.py | grep -v grep

# 监控日志
tail -f aime_training.log

# 应该看到:
# - [AIMEEvaluator] Loaded 30 AIME problems
# - [DeepWorkflowEnv] Dataset: AIME
# - Epoch 1/50
```

---

## 📊 监控训练

### 查看Pass@K分数
```bash
grep "Pass@K" aime_training.log | tail -20
```

### 查看训练进度
```bash
grep "Epoch\|Episode" aime_training.log | tail -10
```

### 查看GPU使用
```bash
nvidia-smi
```

### 停止训练（如需要）
```bash
kill $(ps aux | grep 'deep_train_real_workflow.py --config aime' | grep -v grep | awk '{print $2}')
```

---

## ⏱️ 预期训练时间

**配置:** aime_config.yaml
- **数据集:** 30个AIME问题（24训练，6测试）
- **Epochs:** 50
- **Episodes per epoch:** 10  
- **Sample size:** 24问题/episode

**时间估算:**
- 每个问题: ~1.5分钟（无WebSearch）
- 每个episode: 24题 × 1.5分钟 = **36分钟**
- 每个epoch: 10 episodes × 36分钟 = **6小时**
- 总训练时间: 50 epochs × 6小时 = **300小时 ≈ 12.5天**

---

## 🔧 故障排查

### 问题1: Qwen模型下载失败
```bash
# 手动下载
pip3 install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir /root/models/Qwen2.5-7B-Instruct
```

### 问题2: 训练启动失败
```bash
# 检查错误日志
tail -100 aime_training.log | grep -i error

# 检查依赖
pip3 install torch transformers datasets pyyaml numpy
```

### 问题3: GPU不可用
```bash
# 检查GPU
nvidia-smi

# 检查CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## 📝 关键文件说明

### aime_config.yaml
- **修改项:** 50 epochs（对应小数据集）
- **优化模型:** GPT-4o（强推理能力）
- **执行模型:** GPT-4o (temperature=0.3)
- **无WebSearch**（按您要求）

### deep_workflow_env.py
- **Line 165:** 传递dataset_type到parser  
- **Line 80-84:** 使用动态dataset而非硬编码

### workflow_parser.py
- **Line 168-248:** 根据dataset类型生成不同interface
- **AIME:** entry_point为Optional[str] = None
- **HumanEval:** entry_point为required str

---

## ✅ 检查清单

部署前检查：
- [x] 目录结构创建
- [x] 集成代码上传
- [x] AFlow代码下载
- [x] AIME数据集下载
- [ ] Qwen模型下载（进行中）

启动前检查：
- [ ] Qwen模型完整下载（15GB）
- [ ] GPU可用（nvidia-smi）
- [ ] Python依赖完整
- [ ] PYTHONPATH设置正确

---

## 🎯 下一步

1. **等待Qwen模型下载完成**（15-30分钟）
   ```bash
   tail -f /root/qwen_download.log
   ```

2. **下载完成后启动训练**
   ```bash
   cd /root/integration
   export PYTHONPATH=/root/AFlow:/root/integration:$PYTHONPATH
   nohup python3 deep_train_real_workflow.py --config aime_config.yaml > aime_training.log 2>&1 &
   ```

3. **监控训练进度**
   ```bash
   tail -f aime_training.log
   ```

---

**部署完成日期:** 2025-10-16  
**预计可启动时间:** 下载完成后立即可用  
**预计训练完成时间:** 启动后约12.5天  

---
**状态:** ✅ **就绪 - 等待Qwen模型下载完成**
