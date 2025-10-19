#!/bin/bash
# 消融实验监控脚本

echo "监控消融实验进度..."
echo "服务器: root@0.tcp.ngrok.io:11729"
echo ""

while true; do
    clear
    echo "========================================"
    echo "  消融实验实时监控"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    echo ""

    # 检查进程
    sshpass -p 'MLUerV93OMJH' ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p 11729 root@0.tcp.ngrok.io "
        cd /root/integration

        # 进程状态
        echo '📊 进程状态:'
        ps aux | grep 'ablation_study.py' | grep -v grep || echo '  ❌ 进程已结束'
        echo ''

        # 日志大小
        if [ -f ablation_study.log ]; then
            echo '📁 日志文件大小:'
            ls -lh ablation_study.log | awk '{print \"  \", \$5, \$9}'
            echo ''
        fi

        # 最新进度
        echo '📝 最新进度 (最后30行):'
        tail -30 ablation_study.log 2>/dev/null || echo '  日志文件未找到'

        # 检查是否完成
        if grep -q '消融实验完成' ablation_study.log 2>/dev/null; then
            echo ''
            echo '✅ ====== 实验完成！ ======'
            echo ''
            grep -A 20 '消融实验总结' ablation_study.log
            exit 0
        fi
    " 2>/dev/null

    echo ""
    echo "========================================"
    echo "按 Ctrl+C 停止监控"
    echo "自动刷新: 30秒"
    echo "========================================"

    sleep 30
done
