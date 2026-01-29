!/bin/bash
# 切换到项目根目录（非常关键）
cd "$(dirname "$0")/../.."

sh scripts/StaTS/ILI.sh
sh scripts/StaTS/ETTh1.sh
sh scripts/StaTS/ETTh2.sh
sh scripts/StaTS/ETTm1.sh
sh scripts/StaTS/ETTm2.sh
sh scripts/StaTS/Electricity.sh
sh scripts/StaTS/Traffic.sh
sh scripts/StaTS/SolarEnergy.sh
