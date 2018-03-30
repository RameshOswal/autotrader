from tweetfeels import TweetFeels
from threading import Thread
import time

#seconds = 10

companies = ['apple', 'microsoft', 'google', 'facebook', 'amazon' ]
keywords = {}

keywords['apple'] = ['apple', 'aapl']
keywords['microsoft'] = ['microsoft', 'msft']
keywords['google'] = ['google', 'googl']
keywords['facebook'] = ['facebook', 'fb']
keywords['amazon'] = ['amazon', 'amzn']

lines_per_file = 10000
def print_sentiments(feels, company, seconds=10):
	i = 0
	filename = company
	file = open(filename + str(int(i/lines_per_file)), "w+");
	while True:
		if (i % lines_per_file == 0):
			file.close();
			file = open(filename + str(int(i/lines_per_file)), "w+");

		time.sleep(seconds)
		if feels.sentiment is not None:
			print('[{}] [{}] Sentiment Score: {}'.format(time.ctime(), company, feels.sentiment.value))
			file.write('[{}] [{}] Sentiment Score: {}\n'.format(time.ctime(), company, feels.sentiment.value))
			i = i + 1


def oath():
	consumer_key = 'S1CmRZwfddAYvirUTro734TZZ'
	consumer_secret = 'O3hn5EoSdZj7LBoC27I2fOLemZz88kqyo9b27wIQU9w4ojyy98'
	access_token = '134553563-A9bsI8PGWegcYf7ImEyjIlDN6tbZgjwBLeTGTD77'
	access_token_secret = 'sboCRhl8W6sQxqq4RHnnTKo3x8uBM2w7ckza8VD5hONMU'

	login = [consumer_key, consumer_secret, access_token, access_token_secret]
	return login

def launch(company, login):
	dbname = company + '.sqlite'
	feels = TweetFeels(login, tracking=keywords[company], db=dbname)
	t = Thread(target=print_sentiments, args=[feels, company])
	feels.start()
	t.start()

def main():
	login = oath()

	#bottom code doesn't work
	'''
	for company in companies:
		launch(company, login)
	'''

	company = 'apple'
	dbname = company + '.sqlite'
	feels = TweetFeels(login, tracking=keywords[company], db=dbname)
	t = Thread(target=print_sentiments, args=[feels, company])

	company = 'microsoft'
	dbname = company + '.sqlite'
	feels2 = TweetFeels(login, tracking=keywords[company], db=dbname)
	t2 = Thread(target=print_sentiments, args=[feels2, company])

	company = 'google'
	dbname = company + '.sqlite'
	feels3 = TweetFeels(login, tracking=keywords[company], db=dbname)
	t3 = Thread(target=print_sentiments, args=[feels3, company])

	company = 'facebook'
	dbname = company + '.sqlite'
	feels4 = TweetFeels(login, tracking=keywords[company], db=dbname)
	t4 = Thread(target=print_sentiments, args=[feels4, company])

	company = 'amazon'
	dbname = company + '.sqlite'
	feels5 = TweetFeels(login, tracking=keywords[company], db=dbname)
	t5 = Thread(target=print_sentiments, args=[feels5, company])


	feels.start()
	feels2.start()
	feels3.start()
	feels4.start()
	feels5.start()
	t.start()
	t2.start()
	t3.start()
	t4.start()
	t5.start()
main()





