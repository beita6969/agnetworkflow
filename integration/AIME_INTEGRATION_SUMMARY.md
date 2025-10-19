# AIME 2024 Integration Summary
## Dataset Migration & Web Search Integration

**Date**: 2025-10-15
**Server**: A100 GPU (root@0.tcp.ngrok.io:11729)
**Status**: ✅ **READY FOR TESTING & TRAINING**

---

## ✅ Completed Tasks

### 1. Dataset Download & Preparation ✅
**AIME 2024 Dataset** successfully downloaded from Hugging Face:
- **Source**: Maxwell-Jia/AIME_2024
- **Size**: 30 problems (24 train, 6 test)
- **Format**: JSONL (JSON Lines)
- **Location**: `/root/AFlow/data/AIME_2024.jsonl` (49KB)
- **Content**: High-difficulty mathematics competition problems
- **Answer Format**: Integers 0-999

**Sample Problem**:
```
Task ID: 2024-II-4
Problem: Let $x,y$ and $z$ be positive real numbers that satisfy system of logarithmic equations...
Answer: 33
```

### 2. New Components Created ✅

#### A. AIME Evaluator (`aime_evaluator.py` - 6.9KB)
**Purpose**: Evaluate workflow performance on mathematical problems

**Key Features**:
- Loads AIME dataset from JSONL
- Extracts numerical answers from natural language responses
- Supports multiple answer formats: `\boxed{42}`, "answer is 42", etc.
- Compares numerical answers for correctness
- Calculates Pass@K metrics

**API Compatibility**: Drop-in replacement for HumanEval evaluator

#### B. WebSearch Operator (`aime_web_search_operator.py` - 4.8KB)
**Purpose**: Enable workflows to search for mathematical knowledge

**Key Features**:
- DuckDuckGo HTML search (no API key required)
- Searches for theorems, formulas, solution methods
- LLM-powered result synthesis
- Timeout handling and error recovery
- Returns concise mathematical knowledge summaries

**Usage in Workflow**:
```python
search_result = await web_search(
    query="logarithm properties",
    problem_context="system of equations"
)
```

#### C. AIME Prompt Manager (`aime_prompt_manager.py` - 4.5KB)
**Purpose**: Generate prompts optimized for mathematical reasoning

**Key Features**:
- Optimization prompts for RL-based workflow improvement
- Execution prompts for problem solving
- Search query generation prompts
- Emphasizes multi-step reasoning and answer validation

#### D. AIME Configuration (`aime_config.yaml` - 3.3KB)
**Purpose**: Training configuration for AIME dataset

**Key Settings**:
- **Dataset**: AIME (30 problems, 24 train / 6 test)
- **Epochs**: 50 (more epochs for smaller dataset)
- **Episodes per epoch**: 10
- **Sample size**: 24 (all training problems)
- **Execution Model**: GPT-4o (stronger reasoning for math)
- **Temperature**: 0.3 (lower for deterministic math)
- **Available Operators**: Custom, MathSolver, WebSearch, ScEnsemble, AnswerExtract

---

## 📊 Key Differences: HumanEval vs. AIME

| Aspect | HumanEval | AIME 2024 |
|--------|-----------|-----------|
| **Task Type** | Code generation | Mathematical reasoning |
| **Dataset Size** | 164 problems | 30 problems |
| **Training Set** | 131 problems (80%) | 24 problems (80%) |
| **Test Set** | 33 problems (20%) | 6 problems (20%) |
| **Evaluation** | Code execution (`exec()`) | Numerical answer extraction |
| **Answer Format** | Code correctness | Integer 0-999 |
| **Difficulty** | Medium (coding) | Very High (competition math) |
| **Required Model** | GPT-4o-mini sufficient | GPT-4o recommended |
| **Web Search** | Not needed | **Highly beneficial** |
| **Typical Pass@K** | 95-99% (with ensemble) | 20-40% expected (very hard) |

---

## 🔧 Integration Architecture

### File Structure
```
/root/integration/
├── aime_evaluator.py           # Math problem evaluator
├── aime_web_search_operator.py # Web search capability
├── aime_prompt_manager.py      # AIME-specific prompts
├── aime_config.yaml            # Training configuration
└── deep_train_real_workflow.py # Training script (reusable)

/root/AFlow/data/
└── AIME_2024.jsonl             # Dataset (30 problems)
```

### How Components Work Together

1. **Training Loop**:
   ```
   Config → Trainer → Environment → Workflow → Evaluator → Reward
   ```

2. **Workflow Execution**:
   ```
   Problem → [WebSearch?] → Math Solver → Answer Extract → Check Answer
   ```

3. **RL Optimization**:
   ```
   Low Reward → Policy Update → Better Workflow Description → Higher Reward
   ```

---

## 🚀 How to Start Training

### Option 1: Quick Test (2-3 problems)
```bash
ssh root@0.tcp.ngrok.io -p 11729
cd /root/integration

# Create test config (sample=2 for quick test)
cp aime_config.yaml aime_config_test.yaml
sed -i 's/sample: 24/sample: 2/' aime_config_test.yaml
sed -i 's/episodes_per_epoch: 10/episodes_per_epoch: 2/' aime_config_test.yaml

# Start test training
export PYTHONPATH=/root/AFlow:/root/integration:/root/verl-agent:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH

nohup python3 deep_train_real_workflow.py \
  --config aime_config_test.yaml \
  > aime_test_training.log 2>&1 &

# Monitor
tail -f aime_test_training.log
```

### Option 2: Full Training (24 problems)
```bash
ssh root@0.tcp.ngrok.io -p 11729
cd /root/integration

export PYTHONPATH=/root/AFlow:/root/integration:/root/verl-agent:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH

nohup python3 deep_train_real_workflow.py \
  --config aime_config.yaml \
  > aime_full_training.log 2>&1 &

# Monitor
tail -f aime_full_training.log
```

---

## ⏱️ Estimated Training Time

### Quick Test (sample=2, 2 episodes/epoch, 10 epochs)
- **Per problem**: ~2-3 minutes (with WebSearch + GPT-4o)
- **Per episode**: ~4-6 minutes (2 problems)
- **Per epoch**: ~8-12 minutes (2 episodes)
- **Total**: ~1.5-2 hours (10 epochs)

### Full Training (sample=24, 10 episodes/epoch, 50 epochs)
- **Per problem**: ~2-3 minutes
- **Per episode**: ~50-75 minutes (24 problems)
- **Per epoch**: ~8.3-12.5 hours (10 episodes)
- **Total**: ~415-625 hours = **17-26 days** (50 epochs)

**Note**: AIME problems are MUCH harder than HumanEval:
- Require deep mathematical reasoning
- May need multiple WebSearch calls
- GPT-4o is slower than GPT-4o-mini
- Expected Pass@K: 20-40% (vs HumanEval's 95-99%)

---

## 🎯 Success Criteria

### Dataset Loading ✅
- [x] Downloaded 30 AIME problems
- [x] Saved in correct JSONL format
- [x] File location: /root/AFlow/data/AIME_2024.jsonl
- [ ] Training log shows "Loaded 30 problems" (to be verified)
- [ ] Training log shows "Testing on 24 problems" (to be verified)

### WebSearch Functionality
- [ ] WebSearch operator loads successfully
- [ ] Can perform DuckDuckGo searches
- [ ] Returns relevant mathematical knowledge
- [ ] Workflows incorporate search results

### Answer Extraction
- [ ] Correctly extracts answers from `\boxed{N}` format
- [ ] Handles "answer is N" format
- [ ] Extracts final numbers from reasoning
- [ ] Numerical comparison works correctly

### Training Progress
- [ ] Pass@K > 0.0 (not dummy data)
- [ ] Pass@K increases over epochs
- [ ] Workflows learn to use WebSearch strategically
- [ ] Final Pass@K > baseline (>15-20%)

---

## 📝 What Changed from HumanEval

### 1. **Evaluation Method**
- **Before**: Execute code with `exec()` and test cases
- **After**: Extract numerical answer and compare integers

### 2. **Workflow Operators**
- **Before**: CustomCodeGenerate, Test, ScEnsemble
- **After**: MathSolver, WebSearch, AnswerExtract, ScEnsemble

### 3. **LLM Selection**
- **Before**: GPT-4o-mini (sufficient for coding)
- **After**: GPT-4o (stronger reasoning for hard math)

### 4. **Temperature**
- **Before**: 0.7 (some creativity for coding)
- **After**: 0.3 (deterministic for mathematics)

### 5. **Expected Accuracy**
- **Before**: 95-99% (HumanEval is relatively easy)
- **After**: 20-40% (AIME is competition-level hard)

---

## 🔍 Monitoring Training

### Check Dataset Loading
```bash
grep "Loaded.*problems" aime_full_training.log
# Should see: [AIMEEvaluator] Loaded 30 AIME problems
```

### Check Training Progress
```bash
grep "Pass@" aime_full_training.log | tail -20
# Should see increasing Pass@K scores
```

### Check WebSearch Usage
```bash
grep -i "websearch\|search" aime_full_training.log | head -20
# Should see WebSearch operator being used
```

### Check GPU Utilization
```bash
nvidia-smi
# Should show ~80-90% GPU utilization during training
```

---

## 💡 Next Steps

### Immediate (Testing Phase)
1. **Run Quick Test**:
   - Start test training with 2 problems
   - Verify all components work correctly
   - Check WebSearch functionality
   - Validate answer extraction

2. **Debug if Needed**:
   - Check logs for any errors
   - Verify AIME dataset loading
   - Test WebSearch operator manually
   - Validate answer parsing logic

### Short-term (If Test Successful)
3. **Start Full Training**:
   - Launch full training with 24 problems
   - Monitor first few episodes carefully
   - Track Pass@K scores
   - Estimate actual completion time

4. **Periodic Monitoring**:
   - Check training progress every 6-12 hours
   - Record Pass@K scores
   - Monitor for errors or stalls
   - Adjust configuration if needed

### Long-term (After Training)
5. **Evaluation**:
   - Test final model on all 30 problems
   - Compare train vs test set performance
   - Analyze which problem types improved most
   - Assess WebSearch effectiveness

6. **Analysis**:
   - Which workflows performed best?
   - How often was WebSearch used?
   - Did ensemble methods help?
   - What's the learning curve shape?

---

## 🎉 Summary

### What We've Accomplished

1. ✅ **Downloaded AIME 2024 dataset** (30 high-difficulty math problems)
2. ✅ **Created AIME Evaluator** (extracts and validates numerical answers)
3. ✅ **Implemented WebSearch Operator** (searches for mathematical knowledge)
4. ✅ **Built AIME Prompt Manager** (math-specific prompting)
5. ✅ **Configured AIME Training** (optimized for small, hard dataset)
6. ✅ **Uploaded All Files to Server** (ready for execution)

### Current State

- **Dataset**: ✅ Ready (30 problems, 49KB JSONL file)
- **Code**: ✅ Ready (4 new Python files, 1 config file)
- **Server**: ✅ Ready (A100 GPU, all dependencies installed)
- **Training**: ⏳ Ready to start (awaiting your command)

### Key Differences from HumanEval

- **Much harder**: AIME is competition-level math (expect 20-40% vs 95-99%)
- **Much slower**: GPT-4o + WebSearch = 2-3 min/problem vs 25 sec/problem
- **Longer training**: 17-26 days vs 3.5 days (smaller dataset but harder/slower)
- **New capability**: WebSearch enables knowledge augmentation
- **Different evaluation**: Numerical answer extraction vs code execution

---

**Status**: ✅ **COMPLETE - READY FOR TRAINING**
**Next Step**: Run quick test to verify all components, then launch full training
**Estimated Full Training Time**: 17-26 days (50 epochs, 24 problems/episode)

---

## 附录：重要命令速查 Quick Command Reference

```bash
# 连接服务器 Connect to server
ssh root@0.tcp.ngrok.io -p 11729  # Password: MLUerV93OMJH

# 设置环境变量 Set environment
export PYTHONPATH=/root/AFlow:/root/integration:/root/verl-agent:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH

# 快速测试 Quick test
python3 deep_train_real_workflow.py --config aime_config_test.yaml

# 完整训练 Full training
nohup python3 deep_train_real_workflow.py --config aime_config.yaml > aime_training.log 2>&1 &

# 监控日志 Monitor log
tail -f aime_training.log

# 检查进程 Check process
ps aux | grep deep_train_real_workflow.py

# 检查GPU Check GPU
nvidia-smi

# 停止训练 Stop training
kill $(ps aux | grep 'deep_train_real_workflow.py --config aime' | grep -v grep | awk '{print $2}')
```
