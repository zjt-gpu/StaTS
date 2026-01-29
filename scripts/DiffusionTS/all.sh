!/bin/bash
# 切换到项目根目录（非常关键）
cd "$(dirname "$0")/../.."

sh scripts/DiffusionTS/Traffic.sh
sh scripts/DiffusionTS/ILI.sh
sh scripts/DiffusionTS/ETTh1.sh
sh scripts/DiffusionTS/ETTh2.sh
sh scripts/DiffusionTS/ETTm1.sh
sh scripts/DiffusionTS/ETTm2.sh
sh scripts/DiffusionTS/Electricity.sh
sh scripts/DiffusionTS/SolarEnergy.sh