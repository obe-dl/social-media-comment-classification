from __future__ import unicode_literals

from django.db import models


class Type(models.Model):
    name = models.CharField(max_length=1024)

    def __unicode__(self):
        return self.name


class Page(models.Model):
    type = models.ForeignKey(Type)
    name = models.CharField(max_length=1024)
    scrapped = models.BooleanField(default=False)

    def __unicode__(self):
        return '{t} {n} {s}'.format(t=self.type,
                                    n=self.name,
                                    s=self.scrapped)


class Post(models.Model):
    page = models.ForeignKey(Page)
    message = models.TextField(blank=True, null=True)
    post_id = models.CharField(max_length=1024)
    created_time = models.CharField(max_length=1024)
    json = models.TextField()
    scrapped = models.BooleanField(default=False)

    def __unicode__(self):
        return '{s}'.format(s=self.scrapped)


class Comment(models.Model):
    post = models.ForeignKey(Post)
    message = models.TextField(blank=True, null=True)
    comment_id = models.CharField(max_length=1024)
    json = models.TextField()
    created_time = models.CharField(max_length=1024)

    def __unicode__(self):
        return '{t} {m}'.format(t=self.post.page.type, m=self.message)
