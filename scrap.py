import requests
import threading
import os
from datetime import datetime
import math
import time
import csv
import multiprocessing
from random import randrange

def get_markets():
    markets_temp = requests.get('https://www.binance.com/fapi/v1/exchangeInfo')
    markets_temp = markets_temp.json()
    mk = []
    for i in range(len(markets_temp['symbols'])):
        mk.append(markets_temp['symbols'][i]['symbol'])
    return mk


def store(market:str):
    print(market)
    endTime = 1000*int(math.floor(time.time()))
    counter = 0
    result = []
    while True:
        t = []
        histPrice_postfix = 'symbol=' + market + '&interval=2h&limit=' + str(480) + "&endTime=" + str(endTime)
        # print(histPrice_postfix)
        hs = requests.get('https://www.binance.com/fapi/v1/klines',histPrice_postfix).json()
        for i in hs:
            del i[-1]
            del i[6]
            del i[6]
            # print(i)
        hs.reverse()
        lsur_postfix = 'symbol=' + market + '&period=2h&limit='+ str(480) + "&start=" + str(endTime)
        lsur = requests.get('https://www.binance.com/futures/data/globalLongShortAccountRatio',lsur_postfix).json()
        lsur.reverse()

        open_interest_postfix = 'symbol=' + market + '&period=2h&limit='+ str(480) + "&start=" + str(endTime)
        oI = requests.get('https://www.binance.com/futures/data/openInterestHist',open_interest_postfix).json()
        oI.reverse()
        # print(market,len(hs))
        if(len(hs)<480):
            break
        for i,j,k in zip(hs,lsur,oI):
            f = []
            f+=i[:-3]
            f+=[j['longAccount']]
            f+=[j['shortAccount']]
            f+=[j['longShortRatio']]
            f+=[k['sumOpenInterest']]
            t.append(list(f))
        endTime = hs[479][0] - 7200000
        counter += 1
        result += t
        del hs,t
        # print(market,counter)
        # time.sleep(0.1)
    result.append(["date","open","high","low","close","volume","longAccount","shortAccount","longShortRatio","sumOpenInterest"])
    result.reverse()
    for i in result:
        try:
            t = datetime.fromtimestamp(i[0]/1000)
            i[0] = t
        except:
            continue

    # exit()
    if(len(result)<2):
        return
    with open('./future_hist_2h/'+market+".csv", "w", newline="") as f:
        writer = csv.writer(f)
        
        writer.writerows(result)

def bridge(mk:list):
    for i in mk:
        store(i)

if __name__ == '__main__':
    mk = get_markets()
    mk.sort()
    print(len(mk))
    for i in mk:
        store(i)
    processes = []
    p1 = multiprocessing.Process(target = bridge, args = (mk[:30:],))
    p2 = multiprocessing.Process(target = bridge, args = (mk[30:60:],))
    p3 = multiprocessing.Process(target = bridge, args = (mk[60:90:],))
    p4 = multiprocessing.Process(target = bridge, args = (mk[90::],))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    # store("BTCUSDT")