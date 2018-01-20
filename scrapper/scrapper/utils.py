from datetime import datetime
from json import loads, dumps
from time import sleep
from urllib2 import urlopen, Request

import unidecode
from conf import APP_ID, APP_SECRET

from scrapper.models import Page, Post, Comment

RANGES = [('2017-01-01', '2018-01-01'),
          ('2016-01-01', '2017-01-01'),
          ('2015-01-01', '2016-01-01'),
          ('2014-01-01', '2015-01-01'),
          ('2013-01-01', '2014-01-01'),
          ('2012-01-01', '2013-01-01'),
          ('2011-01-01', '2012-01-01'),
          ('2010-01-01', '2011-01-01'),
          ]

ACCESS_TOKEN = APP_ID + "|" + APP_SECRET

POST_URL = 'https://graph.facebook.com/v2.9/{page_id}/posts/?access_token={access_token}&since={since}&until={until}'
COMMENT_URL = 'https://graph.facebook.com/v2.9/{post_id}/comments/?access_token={access_token}'


def _scrape_post(access_token, post):
    request = Request(COMMENT_URL.format(post_id=post.post_id, access_token=access_token))
    while True:
        count = 5
        while count:
            try:
                response = urlopen(request)
                if response.getcode() == 200:
                    break
            except:
                sleep(5)
                count -= 1
        if not count:
            return
        comments = loads(response.read())
        for comment in comments['data']:
            Comment(post=post,
                    message=unidecode.unidecode(comment.get('message', u'')),
                    comment_id=comment['id'],
                    json=dumps(comment)).save()

        if 'paging' in comments:
            request = Request(
                COMMENT_URL.format(post_id=post.post_id, access_token=access_token) + '&after={after}'.format(
                    after=comments['paging']['cursors']['after']))
        else:
            break


def _scrape_page(access_token, page, start, end):
    print page.name, start, end, datetime.utcnow()
    request = Request(POST_URL.format(page_id=page.name, access_token=access_token, since=start, until=end))
    while True:
        while True:
            try:
                response = urlopen(request)
                if response.getcode() == 200:
                    break
            except:
                sleep(5)
        posts = loads(response.read())
        for post in posts['data']:
            Post(page=page,
                 message=unidecode.unidecode(post.get('message', u'')),
                 post_id=post['id'],
                 created_time=post['created_time'],
                 json=dumps(post),
                 ).save()

        if 'paging' in posts:
            request = Request(
                POST_URL.format(page_id=page.name, access_token=access_token, since=start,
                                until=end) + '&after={after}'.format(
                    after=posts['paging']['cursors']['after']))
        else:
            break


def scrape_page_posts(page_name, ranges=None):
    page = Page.objects.get(name=page_name, scrapped=False)
    if ranges is None:
        ranges = RANGES
    for range in ranges:
        _scrape_page(ACCESS_TOKEN, page, range[0], range[1])
    page.scrapped = True
    page.save()


def scrape_page_comments(page_name):
    posts = Post.objects.filter(page__name=page_name, scrapped=False).order_by('id')
    print 'Scrapping {c} posts'.format(c=posts.count()), datetime.utcnow()
    for post in posts:
        print post.id, datetime.utcnow()
        _scrape_post(ACCESS_TOKEN, post)
        post.scrapped = True
        post.save()


def normalize(m):
    n = ''
    m = m.lower()
    for i in m:
        if ord('a') <= ord(i) <= ord('z') or i in [' ', '.']:
            n += i
    return n


def check(m):
    for i in m:
        if ord('a') <= ord(i) <= ord('z'):
            return True


def dump_it(type_name):
    comments = Comment.objects.filter(post__page__type__name=type_name).values('comment_id', 'message')[:130000]

    with open('/tmp/{t}'.format(t=type_name), 'w') as f:
        for comment in comments:
            if check(normalize(comment['message'])):
                f.write('{m}\n'.format(i=comment['comment_id'], m=normalize(comment['message'])))
