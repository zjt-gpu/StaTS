!/bin/bash
# 切换到项目根目录（非常关键）
cd "$(dirname "$0")/../.."

sh scripts/NSDiff/Electricity.sh
sh scripts/NSDiff/Traffic.sh
sh scripts/NSDiff/ETTh1.sh
sh scripts/NSDiff/ETTh2.sh
sh scripts/NSDiff/ETTm1.sh
sh scripts/NSDiff/ETTm2.sh
sh scripts/NSDiff/ILI.sh
sh scripts/NSDiff/SolarEnergy.sh
