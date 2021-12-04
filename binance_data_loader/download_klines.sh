#!/bin/bash

symbols=("BTCUSDT") # add symbols here to download
# intervals=("1m" "5m" "15m" "30m" "1h" "2h" "4h" "6h" "8h" "12h" "1d" "3d" "1w" "1mo")
intervals=("15m")
years=("2017" "2018" "2019" "2020")
months=(01 02 03 04 05 06 07 08 09 10 11 12)

baseurl="https://data.binance.vision/data/spot/monthly/klines"

for symbol in ${symbols[@]}; do
  for interval in ${intervals[@]}; do
    for year in ${years[@]}; do
      for month in ${months[@]}; do
        url="${baseurl}/${symbol}/${interval}/${symbol}-${interval}-${year}-${month}.zip"
        response=$(wget --server-response -q ${url} 2>&1 | awk 'NR==1{print $2}')
        if [ ${response} == '404' ]; then
          echo "File not exist: ${url}" 
        else
        #   echo "downloaded: ${url}"
          $(mkdir -p ./raw_data/${symbol})
          $(cd ./raw_data/${symbol} && curl -LO -s ${url})
          $(cd ./raw_data/${symbol} && unzip ${symbol}-${interval}-${year}-${month}.zip)
          $(cd ./raw_data/${symbol} && rm ${symbol}-${interval}-${year}-${month}.zip)
        fi
      done
    done
  done
done 