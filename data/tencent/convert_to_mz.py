#!/usr/bin/env python
# coding=utf-8
# convert to the dataformat of matchzoo

import jieba
with open('sample.tr','rb') as fin:
    with open("output_mz",'w') as fout:
        for e,line in enumerate(fin):
            print(e)
            fields = line.strip().split("\001")
            query = ' '.join(list(jieba.cut(fields[0]))).encode('utf-8')
            app = ' '.join(list(jieba.cut(fields[1]))).encode('utf-8')
            fout.write('%s\t%s\t%s\n' % (query,app,str(fields[2])))
