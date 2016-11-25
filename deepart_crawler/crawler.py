#!/usr/bin/python

import os, threading, time
import urllib, ssl
import HTMLParser
from Queue import Queue


class MyHTMLParser(HTMLParser.HTMLParser):
    callback = None

    def __init__(self, callback):
        HTMLParser.HTMLParser.__init__(self)
        self.callback = callback

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'img':
            self.callback.accept(attrs['src'])
        if tag == 'div' and 'class' in attrs and \
                attrs['class'] == "col-md-4 col-sm-6 col-xs-12":
            self.callback.accept(attrs['id'])


class DownloadThread(threading.Thread):  # a worker
    def __init__(self, queue):
        super(DownloadThread, self).__init__()
        self.queue = queue
        self.daemon = True

    def run(self):
        while True:
            url, dest_path = self.queue.get()
            try:
                urllib.urlretrieve(url, dest_path)
            except Exception, e:
                print "   Error: %s" % e
            self.queue.task_done()


class Downloader:  # a state machine, and a master
    STATES = ['ID', 'G', 'C', 'S'] # id, generated img, content img, style img respectively
    state = 0
    last_id = None

    MAX_THREADS = 4
    BLOCK_THRESHOLD = 20  # Block when too many downloads haven't been finished
    queue = Queue(maxsize=BLOCK_THRESHOLD)
    threads = [DownloadThread(queue) for i in xrange(MAX_THREADS)]

    def __init__(self, out_dir):
        self.out_dir = out_dir
        for t in self.threads:
            t.start()

    def accept(self, str):
        print self.STATES[self.state] + " : " + str

        if self.STATES[self.state] == 'ID':
            self.last_id = str
        else:
            _, ext = os.path.splitext(str)
            url = str
            dest_path = '%s/%s_%s%s' % (self.out_dir, self.last_id, self.STATES[self.state], ext)
            self.queue.put((url, dest_path))

        self.state = (self.state + 1) % len(self.STATES)


def main():

    # some initialization

    out_dir = './downloaded'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    downloader = Downloader(out_dir='downloaded')
    parser = MyHTMLParser(downloader)


    # start

    url = r'https://deepart.io/latest/?last=P3gFOtad'

    try:
        while True:
            # check exit condition
            if os.path.exists(out_dir+'/STOP!'):
                os.remove(out_dir+'/STOP!')
                raise(EnvironmentError("STOP signal detected."))

            filehandle = urllib.urlopen(url, context=ssl.SSLContext(ssl.PROTOCOL_TLSv1))
            html = filehandle.read()
            parser.feed(html)

            print ('Finished extracting image url. Waiting for download threads to finish...')
            downloader.queue.join()  # wait until all images in the queue finished downloading
            print ('Download thread finished. Starting next batch...')

            with open('last_downloaded_id.txt', 'a') as f:
                f.write(downloader.last_id + " " + time.asctime()+'\n')

            url = r'https://deepart.io/latest/?last=' + downloader.last_id

    except Exception as ex:
        print "Exception occured: " + repr(ex) + " Last downloaded ID:" + downloader.last_id



testHTML = """
                <div class="col-md-4 col-sm-6 col-xs-12" id="IA1Cv4Dc1">
                    <div class="artwork" style="margin-bottom: 15px;">
                        <img class="artwork-img img-responsive" src="https://deepart-io.s3.amazonaws.com/cache/3e/12/3e12408f0e9c9eed00abedc47f6d6160.jpg">
                        <a href="/img/IA1Cv4Dc1/" class="artwork-controls">
                            <p class="feedback" id="num-likesIA1Cv4Dc1">




<span class="glyphicon glyphicon-heart-empty"></span><span class="num-likes"> 21</span><span class="like-text">like</span>
                            </p>
                            <div class="controls artist-info">
                                <p><span class="glyphicon glyphicon-user"></span> Anonymous artist</p>
                            </div>
                        </a>
                        <ul class="list-unstyled artwork-actions">
                            <li><a href="/hire/IA1Cv4Dc1/"><span class="glyphicon glyphicon-picture"></span>Use style</a></li>
                            <li><button onclick="like('IA1Cv4Dc1')" class="btn btn-link btn-like" id="btn-likesIA1Cv4Dc1">



<span class="glyphicon glyphicon-heart-empty"></span><span class="num-likes"> 21</span><span class="like-text">like</span></button></li>
                            <li><a href="/img/IA1Cv4Dc1/"><span class="glyphicon glyphicon-fullscreen"></span>Details</a></li>
                        </ul>
                    </div>
		    <div style="margin-bottom: 45px; text-align: center; vertical-align: middle;">
		        <div style="float: left; width: 50%; text-align: left;">
                        <img class="img-responsive" src="https://deepart-io.s3.amazonaws.com/cache/5e/56/5e5643152aeb12a9cef205e26eaa2826.jpg"/>
			</div>
		        <div style="float: right; width: 50%; text-align: right">
                        <img class="img-responsive" src="https://deepart-io.s3.amazonaws.com/cache/8b/b1/8bb1dff72fd8a6805b7342b02e9bef3a.jpg"/>
			</div>
			<div style="clear: both"></div>
		    </div>
                </div>










                <div class="col-md-4 col-sm-6 col-xs-12" id="69VxYuJs">
                    <div class="artwork" style="margin-bottom: 15px;">
                        <img class="artwork-img img-responsive" src="https://deepart-io.s3.amazonaws.com/cache/f2/ca/f2ca2b91feb77659656f7c8cd6b10fc5.jpg">
                        <a href="/img/69VxYuJs/" class="artwork-controls">
                            <p class="feedback" id="num-likes69VxYuJs">




<span class="glyphicon glyphicon-heart-empty"></span><span class="num-likes"> 32</span><span class="like-text">like</span>
                            </p>
                            <div class="controls artist-info">
                                <p><span class="glyphicon glyphicon-user"></span> Anonymous artist</p>
                            </div>
                        </a>
                        <ul class="list-unstyled artwork-actions">
                            <li><a href="/hire/69VxYuJs/"><span class="glyphicon glyphicon-picture"></span>Use style</a></li>
                            <li><button onclick="like('69VxYuJs')" class="btn btn-link btn-like" id="btn-likes69VxYuJs">



<span class="glyphicon glyphicon-heart-empty"></span><span class="num-likes"> 32</span><span class="like-text">like</span></button></li>
                            <li><a href="/img/69VxYuJs/"><span class="glyphicon glyphicon-fullscreen"></span>Details</a></li>
                        </ul>
                    </div>
		    <div style="margin-bottom: 45px; text-align: center; vertical-align: middle;">
		        <div style="float: left; width: 50%; text-align: left;">
                        <img class="img-responsive" src="https://deepart-io.s3.amazonaws.com/cache/62/0c/620cb25d8ea05a963d1ed38aaa829bdd.jpg"/>
			</div>
		        <div style="float: right; width: 50%; text-align: right">
                        <img class="img-responsive" src="https://deepart-io.s3.amazonaws.com/cache/f2/e2/f2e25ab8a24b590befe508a30fd534a4.jpg"/>
			</div>
			<div style="clear: both"></div>
		    </div>
                </div>

"""
if __name__ == "__main__":
    main()

