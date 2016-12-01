#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os, threading, time, re, shutil
import urllib, ssl
import HTMLParser
from Queue import Queue
import requests


class DownloadThread(threading.Thread):  # a worker
    def __init__(self, queue):
        super(DownloadThread, self).__init__()
        self.queue = queue
        self.daemon = True

    def run(self):
        while True:
            url, dest_path = self.queue.get()
            try:
                r = requests.get(url, stream=True, headers={'User-agent': 'Mozilla/5.0'})
                if r.status_code == 200:
                    with open(dest_path, 'wb') as f:
                        r.raw.decode_content = True
                        shutil.copyfileobj(r.raw, f)
            except Exception, e:
                print "   Error: %s" % e
            self.queue.task_done()


class Downloader:  # a state machine, and a master
    # like.png, generated img, content img, style img, waiting for next like.png respectively
    STATES = ['WAIT_FOR_LIKE.PNG', 'LIKE.PNG', 'G', 'C', 'S']
    state = 0
    last_generated_img_ID = None

    MAX_THREADS = 4
    BLOCK_THRESHOLD = 20  # Block when too many downloads haven't been finished
    queue = Queue(maxsize=BLOCK_THRESHOLD)
    threads = [DownloadThread(queue) for i in xrange(MAX_THREADS)]

    def __init__(self, out_dir):
        self.out_dir = out_dir
        for t in self.threads:
            t.start()

    def accept(self, str):
        if str == '/like.png':
            self.state = 1
        else:
            last_s = self.STATES[self.state]
            if last_s == 'LIKE.PNG' or last_s == 'G' or last_s == 'C':
                self.state = (self.state + 1) % len(self.STATES)

                if self.STATES[self.state] == 'G':
                    self.last_generated_img_ID = re.search(r'_(img\d+)_', str).group(1)  # group 0: whole match, 1: () part

                str = str.replace('thumb_', '')
                _, ext = os.path.splitext(str)
                url = str
                dest_path = '%s/%s_%s%s' % (self.out_dir, self.last_generated_img_ID, self.STATES[self.state], ext)
                self.queue.put((url, dest_path))
            else:
                self.state = 0

        print self.STATES[self.state] + " : " + str


class MyHTMLParser(HTMLParser.HTMLParser):
    callback = None

    def __init__(self, callback):
        HTMLParser.HTMLParser.__init__(self)
        self.callback = callback

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'img':
            self.callback.accept(attrs['src'])


def main():

    # some initialization

    out_dir = './downloaded'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    downloader = Downloader(out_dir='downloaded')
    parser = MyHTMLParser(downloader)

    # start crawling
    # To resume crawling, simply search for next url in the log file last_downloaded_id.txt, and copy it here
    url = r'http://ostagram.ru/static_pages/lenta?_=1480121644785&ftime=1044363&last_days=30&locale=en&page=3&ftime=1003738'

    try:
        while True:
            # check exit condition
            if os.path.exists(out_dir+'/STOP!'):
                os.remove(out_dir+'/STOP!')
                raise(EnvironmentError("STOP signal detected."))

            # send request, extract the html to parse and extract the next url to visit

            r = requests.get(url, stream=True,
                             headers={'User-agent': 'Mozilla/5.0', 'X-Requested-With':'XMLHttpRequest',
                                      'Accept': 'text/javascript, application/javascript, application/ecmascript, application/x-ecmascript, */*; q=0.01'
                                      })
            lines = r.text.split('\n')

            html = lines[0][len("$('#posts').append('") : -len("');")]
            html = html.encode('utf-8')
            html = html.replace(r'\/', r"/").replace(r'\"', r'"').replace(r'\n', '\n')

            nexturl = lines[1].replace(r'\/', r"/").replace(r'\"', r'"').replace(r'\n', '\n')
            nexturl = re.search(r'class="next_page" rel=\"next\" href="(.*?)">Next', nexturl).group(1)
            nexturl += "&ftime="
            nexturl += re.search(r'ftime="(\d+)"', lines[2]).group(1)
            nexturl = 'http://ostagram.ru' + nexturl

            # done extracting

            parser.feed(html)

            print ('next url: %s' % nexturl)
            print ('Finished extracting image url. Waiting for download threads to finish...')
            downloader.queue.join()  # wait until all images in the queue finished downloading
            print ('Download thread finished. Starting next batch...')

            with open('last_downloaded_id.txt', 'a') as f:
                f.write(downloader.last_generated_img_ID + " next url: " + nexturl + ' ' + time.asctime()+'\n')

            url = nexturl

    except Exception as ex:
        print "Exception occured: " + repr(ex) + " Last downloaded ID:" + downloader.last_generated_img_ID



testHTML = u"""<div class="row itemRow">
  <div class="col-lg-1 col-md-1 col-sm-2 col-xs-2 clearLRpadding">
    <img class="imagesStyle" src="http://files2.ostagram.ru/uploads/client/avatar/163482/avatar100_img.png" alt="Avatar100 img" />
  </div>

  <div class="col-lg-8 col-md-8 col-sm-4 col-xs-4 ">
    <div class="centerText1">
      <a title="Look all images this author" href="/static_pages/lenta?client_id=163482&amp;locale=en">Андрей</a>
    </div>
  </div>
  <div class="col-lg-1 col-md-1 col-sm-2 col-xs-2 clearLRpadding">
  </div>
  <div class="col-lg-1 col-md-1 col-sm-2 col-xs-2 clearLRpadding">
  </div>

  <div class="col-lg-1 col-md-1 col-sm-2 col-xs-2 clearLRpadding">
    <div class="like_panel_1080422" style="position: relative">
  <div class="centerTextLike1 textCenter"  style="">32</div>

  <div class="absolutePos">
        <a href="/clients/sign_up?locale=en"><img alt="like" class="imagesStyle" src="/like.png" /></a>
  </div>


</div>
  </div>
</div>

<div class="row pos-relative clearLRpadding2" >
      <a href="/clients/sign_up?locale=en"><img class="imagesStyle" src="http://files2.ostagram.ru/uploads/pimage/imageurl/1490764/thumb_img1080422_9a4dd064c55972a3.jpg" alt="Thumb img1080422 9a4dd064c55972a3" /></a>
      <div class="downloadImage1">
      </div>
</div>

<div class="row">
  <div class="col-lg-6 col-md-6 col-sm-6 col-xs-6  clearLRpadding2">
    <a class="imagesStyle" data-lightbox="content" href="http://files2.ostagram.ru/uploads/content/image/1073142/img_1e6889af17.jpg"><img class="imagesStyle" src="http://files2.ostagram.ru/uploads/content/image/1073142/thumb_img_1e6889af17.jpg" alt="Thumb img 1e6889af17" /></a>
  </div>

  <div class="col-lg-6 col-md-6 col-sm-6 col-xs-6  clearLRpadding2">
    <a title="Look all images in this style" href="/static_pages/lenta?locale=en&amp;style_id=287844"><img class="imagesStyle" src="http://files2.ostagram.ru/uploads/style/image/287844/thumb_img_6f5b166469.jpg" alt="Thumb img 6f5b166469" /></a>
  </div>



</div><div class="row itemRow">
  <div class="col-lg-1 col-md-1 col-sm-2 col-xs-2 clearLRpadding">
    <img class="imagesStyle" src="http://files2.ostagram.ru/uploads/client/avatar/24816/avatar100_img.jpg" alt="Avatar100 img" />
  </div>

  <div class="col-lg-8 col-md-8 col-sm-4 col-xs-4 ">
    <div class="centerText1">
      <a title="Look all images this author" href="/static_pages/lenta?client_id=24816&amp;locale=en">Sergey</a>
    </div>
  </div>
  <div class="col-lg-1 col-md-1 col-sm-2 col-xs-2 clearLRpadding">
  </div>
  <div class="col-lg-1 col-md-1 col-sm-2 col-xs-2 clearLRpadding">
  </div>

  <div class="col-lg-1 col-md-1 col-sm-2 col-xs-2 clearLRpadding">
    <div class="like_panel_1032876" style="position: relative">
  <div class="centerTextLike1 textCenter"  style="">32</div>

  <div class="absolutePos">
        <a href="/clients/sign_up?locale=en"><img alt="like" class="imagesStyle" src="/like.png" /></a>
  </div>


</div>
  </div>
</div>

<div class="row pos-relative clearLRpadding2" >
      <a href="/clients/sign_up?locale=en"><img class="imagesStyle" src="http://files2.ostagram.ru/uploads/pimage/imageurl/1441235/thumb_img1032876_ccc33355738cd311.jpg" alt="Thumb img1032876 ccc33355738cd311" /></a>
      <div class="downloadImage1">
      </div>
</div>

<div class="row">
  <div class="col-lg-6 col-md-6 col-sm-6 col-xs-6  clearLRpadding2">
    <a class="imagesStyle" data-lightbox="content" href="http://files2.ostagram.ru/uploads/content/image/1025838/img_e8fbc2f1bd.jpg"><img class="imagesStyle" src="http://files2.ostagram.ru/uploads/content/image/1025838/thumb_img_e8fbc2f1bd.jpg" alt="Thumb img e8fbc2f1bd" /></a>
  </div>

  <div class="col-lg-6 col-md-6 col-sm-6 col-xs-6  clearLRpadding2">
    <a title="Look all images in this style" href="/static_pages/lenta?locale=en&amp;style_id=195404"><img class="imagesStyle" src="http://files2.ostagram.ru/uploads/style/image/195404/thumb_img_8f9fbdb55c.jpg" alt="Thumb img 8f9fbdb55c" /></a>
  </div>
"""
if __name__ == "__main__":
    main()


