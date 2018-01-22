from __future__ import print_function
import datetime

# import base64, hashlib, hmac, time

# timestamp = str(time.time())
# message = timestamp + request.method + request.path_url + (request.body or '')
# hmac_key = base64.b64decode(secret_key)
# signature = hmac.new(hmac_key, message.encode('utf-8'), hashlib.sha256)


# There are 1440 minutes in one day and 7200 in five. This numer can be diveded by 200 which yields 36. 
# Therefore we should grep data in five days batches.

import gdax
import numpy as np
import pandas as pd
import sys
import time


public_client = gdax.PublicClient()
start_time = datetime.datetime(2017, 8, 1, 0, 0, 0) 
end_time = start_time + datetime.timedelta(seconds=200*60)
# next_timeframe = next_timeframe + datetime.timedelta(seconds=200)
end = datetime.datetime(2017, 10, 1, 0, 0, 0).isoformat()

# print('start time:', start)
# print('next_timeframe:', next_timeframe)
# print('end time:', end)

# a = public_client.get_product_historic_rates('ETH-USD',start=start_time.isoformat(), end=end_time.isoformat(), granularity=60)
# print(a)
 
# sys.exit()
requestsPerFile = 36 # eigentlich 36
for interval in range(0,12):

  firstTimeStampInFile = time.mktime(start_time.timetuple())
  b = [None] * requestsPerFile # eigentlich 36
  # sys.exit()
  for i in range(0, requestsPerFile): # eigentlich 0,36
    print('start_time: ', start_time)
    print('end_time:   ', end_time)
    a = public_client.get_product_historic_rates('ETH-USD',start=start_time.isoformat(), end=end_time.isoformat(), granularity=60)
    #print('a', a)
    start_time = start_time + datetime.timedelta(seconds=200*60)
    end_time = start_time + datetime.timedelta(seconds=200*60)
    b[-(i+1)] = a 
    print('len(b[-(i+1)]): ',len(b[-(i+1)]))
    time.sleep(1.5)

  c=[]
  for ii in range(0,len(b)):
  	for jj in range(0,len(b[ii])):
  	  c.append(b[ii][jj])

  # c=[]
  # print('len(b)',len(b))
  # for ii in range(0,len(b)):
  #   if len(b[ii]) <= 1:
  #     print('upsi',b[ii])
  #     print('len(b[ii])',len(b[ii]))
  #     print('Something went terible wrong!')
  #     sys.exit()

  #   for jj in range(0,200):
  #     print(firstTimeStampInFile)
  #     print(firstTimeStampInFile + 60*jj)
  #     try:
  #       if b[ii][jj][0] - firstTimeStampInFile + 60*jj > 60.:
  #     	  c.append(b[ii][jj-1])
  #         c[-1][1] = c[-1][1]+60
  #         c[-1][2] = 'Here!'
  #       elif b[ii][jj][0] - firstTimeStampInFile + 60*jj == 60.:
  #     	  c.append(b[ii][jj])
  #     except:
  #     	print('exception!')
  #       print('len(b[ii])',len(b[ii]))
  #   print(len(c))
  #     # try:
  #     # 	print(b[ii][jj][0])
  #     #   if b[ii][jj][0] - b[ii][jj-1][0] > 60.:
  #     #     c.append(b[ii][jj-1])
  #     #     c[-1][1] = c[-1][1]+60
  #     #     c[-1][2] = 'Here!'
  #     #   else:
  #     #     c.append(b[ii][jj])
  #     # except:
  #     # 	c.append([None, None, None, None, None, None ])

  
  df = pd.DataFrame(c,columns=['time', 'low', 'high', 'open', 'close', 'volume' ])
  # firstTimeStampInFile
  df = df.iloc[::-1]
  df = df.reset_index()
  df = df.drop(['index'], axis=1)
  # print(df)

  d = df.values.tolist()
  print(len(d))

  for ll in range(len(d)-1):
  	if ( d[ll+1][0] - d[ll][0] ) > 60.:
  	  print('d[ll]        ',d[ll])
  	  print('d[ll+1]      ',d[ll+1])
  	  d.insert(ll+1,d[ll])
  	  d[ll+1][0] = d[ll+1][0] + 60
  	  d[ll+1][1] = d[ll+1][4]
  	  d[ll+1][2] = d[ll+1][4]
  	  d[ll+1][3] = d[ll+1][4]
  	  d[ll+1][5] = 0.0
  	  print('d[ll+1] (new)',d[ll+1])

  print(len(d))

  df = pd.DataFrame(d,columns=['time', 'low', 'high', 'open', 'close', 'volume' ])
  print(df)
  # print(df['index'])
  for kk in range( 0,len(df['time']) ):
    if df['time'][kk] - (df['time'][0] + 60*kk) > 60.: #Ab dem ersten eintreffen ist die schleife immer true solange das problem nicht gefixt ist!
      print(df['time'][0])
      print(df['time'][0] + 60*kk)
      print(df['time'][kk])
      print('Not in seq.')

  startOfFile = df.iloc[0,0]
  endOfFile = df.iloc[-1,0]
  # df.to_csv('days_'+str(interval*5+1)+'-'+str(interval*5+5)+'.csv')
  df.to_csv('days_'+str(startOfFile)+'-'+str(endOfFile)+'.csv')
  time.sleep(2.0)

#np.savetxt("foo.csv", b_np, delimiter=",")

# print(start_time.isoformat())
# print(end_time.isoformat())

# print(public_client.get_products())

# b = public_client.get_product_historic_rates('BTC-USD',start=start_time.isoformat(), end=end_time.isoformat(), granularity=60)

# print('len(b):',len(b))
# b_np = np.array(b)
# print(b_np.shape)
# # print(b_np[:,:,0])
# print(len(b_np[0,0]))

# c = []
# for ii in range(len(b_np)):
#   for jj in range(len(b_np[0])):
#     c.append(datetime.datetime.fromtimestamp(int(b_np[ii,jj,0])).strftime('%Y-%m-%d %H:%M:%S'))

# print('c:',c)
# print('len(c):',len(c)