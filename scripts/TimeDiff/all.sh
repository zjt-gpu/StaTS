!/bin/bash
# 切换到项目根目录（非常关键）
cd "$(dirname "$0")/../.."

sh scripts/TimeDiff/Electricity.sh
sh scripts/TimeDiff/Traffic.sh
sh scripts/TimeDiff/Weather.sh
sh scripts/TimeDiff/ETTh1.sh
sh scripts/TimeDiff/ETTh2.sh
sh scripts/TimeDiff/ETTm1.sh
sh scripts/TimeDiff/ETTm2.sh
sh scripts/TimeDiff/ILI.sh
sh scripts/TimeDiff/SolarEnergy.sh
