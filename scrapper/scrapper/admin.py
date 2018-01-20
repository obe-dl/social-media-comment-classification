# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.contrib import admin

from models import Page, Post, Comment, Type

admin.site.register(Type)
admin.site.register(Page)
admin.site.register(Post)
admin.site.register(Comment)
