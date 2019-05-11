import argparse
import os
import time

import requests

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-o', '--output', required=True, help='Path to output directory of images.')
argument_parser.add_argument('-n', '--num-images', type=int, default=1000, help='Number of images to download.')

arguments = vars(argument_parser.parse_args())

url = 'https://www.e-zpassny.com/vector/jcaptcha.do'
total = 0

for i in range(arguments['num_images']):
    try:
        r = requests.get(url, timeout=60)

        p = os.path.sep.join([arguments['output'], f'{str(total).zfill(5)}.jpg'])

        with open(p, 'wb') as f:
            f.write(r.content)
            f.close()

        print(f'[INFO] Downloaded: {p}')
        total += 1
    except:
        print('[INFO] Error downloading image...')

    time.sleep(0.2)  # Don't be a dick with the server
