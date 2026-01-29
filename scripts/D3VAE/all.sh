!/bin/bash
# 切换到项目根目录（非常关键）
cd "$(dirname "$0")/../.."

sh scripts/D3VAE/ILI.sh
sh scripts/D3VAE/ETTm1.sh
sh scripts/D3VAE/ETTm2.sh
sh scripts/D3VAE/Electricity.sh
sh scripts/D3VAE/SolarEnergy.sh
sh scripts/D3VAE/Traffic.sh
sh scripts/D3VAE/ETTh1.sh
sh scripts/D3VAE/ETTh2.sh
