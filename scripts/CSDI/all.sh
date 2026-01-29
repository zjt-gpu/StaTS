!/bin/bash
# 切换到项目根目录（非常关键）
cd "$(dirname "$0")/../.."

sh scripts/CSDI/ILI.sh
sh scripts/CSDI/ETTm1.sh
sh scripts/CSDI/ETTm2.sh
sh scripts/CSDI/Electricity.sh
sh scripts/CSDI/SolarEnergy.sh
sh scripts/CSDI/Traffic.sh
sh scripts/CSDI/ETTh1.sh
sh scripts/CSDI/ETTh2.sh
