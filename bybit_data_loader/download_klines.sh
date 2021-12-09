#!/bin/bash

exchange="ByBit"
symbols=("BTCUSDT") # add symbols here to download
# intervals=("1m" "5m" "15m" "30m" "1h" "2h" "4h" "6h" "8h" "12h" "1d" "3d" "1w" "1mo")
years=("2020")
months=(01 02 03 04 05 06 07 08 09 10 11 12)
dates=(01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)

baseurl="https://public.bybit.com/trading"

for symbol in ${symbols[@]}; do
  for year in ${years[@]}; do
    for month in ${months[@]}; do
      for date in ${dates[@]}; do
        url="${baseurl}/${symbol}/${symbol}${year}-${month}-${date}.csv.gz"
        response=$(wget --server-response -q --spider ${url} 2>&1 | awk 'NR==1{print $2}')
        
        if [[ ${response} == '404' ]]; then
          echo "File not exist: ${url}"
        else
          echo "downloaded: ${url}"
          $(mkdir -p ./raw_data/${exchange})
          $(mkdir -p ./raw_data/${exchange}/${symbol})
          $(cd ./raw_data/${exchange}/${symbol} && curl -LO -s ${url})
          $(cd ./raw_data/${exchange}/${symbol} && gzip -d ${symbol}${year}-${month}-${date}.csv.gz)
          # $(cd ./raw_data/${exchange}/${symbol} && rm ${symbol}${year}-${month}-${date}.csv.gz)
        fi
      done
    done
  done
done