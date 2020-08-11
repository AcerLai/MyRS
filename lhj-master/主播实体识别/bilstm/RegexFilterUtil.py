#!/usr/bin/env python3
# _*_coding:utf-8_*_
#import datetime
#
#__author__ = '刘宇威'
#
#import unicodedata
#import logging


class RegexFilterUtil:
    """  类属性 """
    EMT_PATTERN = r'\[emts\].*?\[/emts\]'  # 懒惰匹配，应对多个表情
    IMG_PATTERN = r'\[img\].*?\[/img\]'  # 懒惰匹配，应对多个图片
    USER_CARD_PATTERN = r'\[userCard.*?](.*?)\[/userCard\]'  # 懒惰匹配, group得到userCard内容
    GROUP_LINK_PATTERN = r"\[\[grouplink\].*?\[/grouplink\]\]"  # 懒惰匹配
    ROOM_LINK_PATTERN = r"\[\[roomlink\].*?\[/roomlink\]\]"
    URL_PATTERN = r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]"
    CHINESE = r"[\u4e00-\u9fa5]"  # 汉字区间
    LATIN = '[' + ''.join([r'\x00-\x7F\x80-\xFF',  # ASCII码： A, À
                           r'\u0100-\u017F',  # 拉丁字符扩展集A： Ā ā
                           r'\u0180-\u024F',  # 拉丁字符扩展集B： ƀ Ɓ
                           r'\u2C60-\u2C7F',  # 拉丁字符扩展集C： Ɑ ⱸ  # 这个区间好像被CC封了
                           r'\uA720-\uA7FF',  # 拉丁字符扩展集D： Ꞓ   # 这个区间好像被CC封了
                           r'\u1E00-\u1EFF',  # 附加拉丁字符扩展集：Ḁḁ
                           ]) + ']'  # LATIN 拉丁字符
    LATIN_LIKE = '[' + ''.join([
                           r'\u0370-\u03FF',  # 希腊文字中的科普特字符（U+0370 – U+03FF）：Μ
                           r'\u0400-\u04FF',  # 西里尔字符（U+0400 – U+04FF）：С К  Х
                           r'\U0001F110-\U0001F1FF',  # Enclosed Alphanumeric Supplement:🄐🄰🅐🇦
                           ]) + ']'  # 类似拉丁字符的Unicode编码
    ARROW_PATTERN = ''.join([
        "←→↔☝⟵⟶⟷⇦⇨⬄⬅⮕➡⬌🡐🡒🡘🠸🠺🡄🡆☟☜☞☚☛👆👇👈👉➤",  # http://xahlee.info/comp/unicode_arrows.html
        'ᐊᐅ',  # 加拿大语字符（U+1400 – U+167F）
    ])

    """" HEART ref: https://unicode-table.com/en/sets/hearts-symbols/ 
                    https://unicode-search.net/unicode-namesearch.pl?term=HEART
        Note：+未添加完
    """
    HEART_PATTERN = 'ღ♥❤💓💕💖💗💙💚💛💜🖤💝❦❧☙❥➳💔💞💟💑💘💌🏩❣♡🧡🎔😍😻'
    PLUS_PATTERN_STR = "[╈╁＋十✚➕┽╬☩✢✣]"  # 加号 \u2548\u2541\uff0b\u5341\u271a\u2795\u253d\u256c

    CN_PUNCTUATION_PATTERN_STR = '|'.join(['[\\u3000-\\u303F]',  # 中文标点符号
                                           '[\\uFE10-\\uFE19]',  # 中文竖排标点
                                           '[\\uFF01-\\uFF0F]',  # 全角ASCII
                                           '[\\uFF1A-\\uFF1F]',  # 全角中英文标点
                                           '[\\uFF3B-\\uFF3F]',  # 半宽片假名、半宽平假名
                                           '[\\uFF5B-\\uFFEE]',  # 半宽韩文字母
                                           '[\\uFE30-\\uFE4F]',  # CJK兼容符号（竖排变体、下划线、顿号）：FE30 - FE4F
                                           '[/]',  # 反斜杠
                                           '[\s]',  # 空白字符
                                           # ''  # 正斜杠TBD
                                           ])
    GAIPIN_PATTERN_STR = "[改該换換用是].{0,5}" \
                         "[拼拚併鉼骿頩渆恲缾剙皏餠帡賆洴蛢鵧姘駢饼誁瓶聠硑胼栟艵軿剏鮩跰郱屏迸絣餅骈荓庰]"
    PINYIN_PATTERN_STR = "[拼拚併鉼骿頩渆恲缾剙皏餠帡賆洴蛢鵧姘駢饼誁瓶聠硑胼栟艵軿剏鮩跰郱屏迸絣餅骈荓庰]" \
                         "[音偣揞窨湆堷愔谙喑腤暗]"
    MAIPIAN_PATTERN_STR = "[卖荬买麦唛迈看瞧][^\\u4e00-\\u9fa5]{0,5}" \
                          "[pP片沜扸偏篇骗翩盘潘兵拼饼姘]"
    QI_E_PATTERN = '|'.join(['企鹅', '启鹅', '启额', '起额', '启额',])
    MA_PATTERN = '|'.join(['码', '🐴', '马', '🐎', ])
    JIAWEIXIN_PATTERN_STR = "([" + ''.join([
                            "%s加嘉珈茄迦]|力口)[^\\u4e00-\\u9fa5]{0,5}" % PLUS_PATTERN_STR.strip('[]'),
                            "([VvṾṼレѵѴ⒱ⓥꝞꝟⱽ微薇嶶威徽徵卫维喂",
                            "俅秋球",
                            "口釦筘抠扣",
                            "寇滱簆蔲蔻冦",
                            "🐧鸟",
                            ]) + ']|' + QI_E_PATTERN + ")"

    """ emoji """
