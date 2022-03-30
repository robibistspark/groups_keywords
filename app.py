import os.path
import requests as rq
from collections import Counter as c
from pymorphy2 import MorphAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from re import sub
import nltk
import numpy as np
import sqlite3
from flask import (Flask, render_template, request, redirect, url_for,
send_file)
import io
from matplotlib import pyplot as plt
import base64
import json

import confid

#%%
con = sqlite3.connect('vk_groups.db')
cur = con.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS groups
(id INTEGER PRIMARY KEY, vk_group text)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS posts
(id INTEGER PRIMARY KEY, post_text text)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS group_to_post
(id INTEGER PRIMARY KEY AUTOINCREMENT, id_group int, id_post int)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS keywords
(id INTEGER PRIMARY KEY, keyword text)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS group_to_keywords
(id INTEGER PRIMARY KEY AUTOINCREMENT, id_group int, id_keyword int, count int)
""")

con.commit()
con.close()

#%%
def get_top_tf_idf_words(tfidf_vector, feature_names, top_n):
    sorted_nzs = np.argsort(tfidf_vector.data)[:-(top_n + 1):-1]
    return feature_names[tfidf_vector.indices[sorted_nzs]]

def parse_group(group_id):
    VERSION = "5.131"
    wall_get_url = "https://api.vk.com/method/wall.get"

    posts = []
    for j in range(0, 201, 100):
        data = rq.get(
            wall_get_url,
            params={
                "owner_id": '-'+str(group_id),
                "count": 100,  # кол-во постов
                "v": VERSION, # версия API
                'offset': j,
                "access_token": confid.TOKEN
            }
        ).json()['response']['items']
        posts.extend([data[i]['text'] for i in range(len(data))])

    posts_preprd = []
    morph = MorphAnalyzer()
    for i in posts:
        i = sub("https://.+\.[a-z]{2,3}", '', i)
        i = sub(" \d+", '', i)
        i = sub("#.+?\b", '', i)
        i = nltk.wordpunct_tokenize(i)
        i = " ".join([morph.parse(j)[0].normal_form for j in i])
        posts_preprd.append(i)

    stops = ['и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между', 'это', 'всё', 'который', 'весь', 'мочь', 'свой', 'ваш', 'твой', 'год', 'также', 'очень']
    tfidf = TfidfVectorizer(stop_words = stops, min_df=3)

    posts_vectors = tfidf.fit_transform(posts_preprd)
    feature_names = np.array(tfidf.get_feature_names())

    keywords_tuples = []
    for i in range(posts_vectors.shape[0]):
        vector = posts_vectors[i, :]
        keywords_tuples.append(tuple(get_top_tf_idf_words(vector,
                                                          feature_names, 5)))

    keywords = []
    for t in keywords_tuples:
        for word in t:
            keywords.append(word)

    group_keywords = list(c(keywords).most_common(5))

    return posts, group_keywords

#%%
app = Flask(__name__)

@app.route('/')
def index():
    topic = 'Получение данных о сообществах ВК'
    return render_template("index.html", topic=topic)

@app.route('/search')
def search_page():
    return render_template('search.html')

#%%

@app.route('/process')
def process():
    if not request.args:
        return redirect(url_for('search_page'))

    group_link = request.args.get('group_link')
    name = os.path.split(group_link)[1]

    utils_resolveScreenName = '''
    https://api.vk.com/method/utils.resolveScreenName'''
    group_id = rq.get(
        utils_resolveScreenName,
        params={
            "screen_name": name,  # ID сообщества
            "v": '5.131', # версия API
            "access_token": confid.TOKEN  # токен доступа
        }
    ).json()['response']['object_id']

    con = sqlite3.connect('vk_groups.db', check_same_thread=False)
    cur = con.cursor()

    # get data into the DB if needed
    cur.execute('SELECT id FROM groups')
    ids_of_groups = [i[0] for i in cur.fetchall()]

    if group_id not in ids_of_groups:
        group_info = parse_group(group_id)

        cur.execute('INSERT INTO groups VALUES (?, ?)', (group_id, name))

        cur.execute('''SELECT id FROM posts
                    ORDER BY id DESC
                    LIMIT 1
                    ''')
        get = cur.fetchall()
        if len(get) > 0:
            last_post_id = get[0][0]
        else:
            last_post_id = 0

        for i in range(len(group_info[0])):
            cur.execute('INSERT INTO posts (id, post_text) VALUES (?, ?)',
                        (last_post_id + i + 1, group_info[0][i]))
            cur.execute('''
                        INSERT INTO group_to_post (id_post, id_group)
                        VALUES (?, ?)''',
                        (last_post_id + i + 1, group_id))

        cur.execute('''SELECT id FROM keywords
            ORDER BY id DESC
            LIMIT 1
            ''')
        get = cur.fetchall()
        if len(get) > 0:
            last_keyword_id = get[0][0]
        else:
            last_keyword_id = 0

        for i in range(len(group_info[1])):
            cur.execute('INSERT INTO keywords (id, keyword) VALUES (?, ?)',
                        (last_keyword_id + i + 1, group_info[1][i][0]))
            cur.execute('''
                        INSERT INTO group_to_keywords (id_group,
                        id_keyword, count) VALUES (?, ?, ?)
                        ''',
                        (group_id, last_keyword_id + i + 1,
                         group_info[1][i][1]))

    con.commit()

    # show plot with keywords
    cur.execute(f'''SELECT keyword, count FROM keywords
        JOIN group_to_keywords ON keywords.id = group_to_keywords.id_keyword
        WHERE group_to_keywords.id_group = {group_id}
        ''')
    kws = cur.fetchall()

    coloring = ['brown', 'green', 'orange', 'plum', 'royalblue']
    for i in range(len(kws)):
        plt.bar([i for i in range(len(kws))],
                [kws[i][1] for i in range(len(kws))],
                color=coloring,
                tick_label=[kws[i][0] for i
                            in range(len(kws))])
        plt.xlabel('Ключевые слова')
        plt.ylabel('Чилсло документов, где слово ключевое')
        plt.title(f'Сообщество {name}')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    data = img.getvalue()     # get data from file (BytesIO)
    data = base64.b64encode(data) # convert to base64 as bytes
    data = data.decode()          # convert bytes to string

    # группы, с которыми есть хотя бы одно общее ключевое слово
    similar = set()
    for i in kws:
        cur.execute(f'''SELECT vk_group FROM keywords
            JOIN group_to_keywords ON keywords.id = group_to_keywords.id_keyword
            JOIN groups ON groups.id = group_to_keywords.id_group
            WHERE keywords.keyword = '{i[0]}' AND groups.id != {group_id}
            ''')
        for j in cur.fetchall():
            if len(similar) > 5:
                break
            if j[0] != name:
                similar.add(j[0])

    con.close()

    return render_template(
        'result.html', similar=similar,
        image='<img src="data:image/png;base64, {}">'.format(data)
        )

#%%
@app.route('/download')
def download():
    return render_template('download.html')

#%%
@app.route('/file')
def file():
    if not request.args:
        return redirect(url_for('search_page'))

    group_link = request.args.get('group_link')
    name = os.path.split(group_link)[1]
    utils_resolveScreenName = '''
    https://api.vk.com/method/utils.resolveScreenName'''

    group_id = rq.get(
        utils_resolveScreenName,
        params={
            "screen_name": name,  # ID сообщества
            "v": '5.131', # версия API
            "access_token": confid.TOKEN  # токен доступа
        }
    ).json()['response']['object_id']

    # with open('processed.txt', 'r', encoding='utf-8') as f:
    #     processed = [i.strip() for i in f.readlines()]

    # if group_id not in processed:
    con = sqlite3.connect('vk_groups.db', check_same_thread=False)
    cur = con.cursor()

    cur.execute('SELECT id FROM groups')
    ids_of_groups = [i[0] for i in cur.fetchall()]
    if group_id not in ids_of_groups:
        return redirect(url_for('search_page'))

    posts_query = f'''
    select post_text from posts
    join group_to_post on posts.id = group_to_post.id_post
    join groups on groups .id = group_to_post.id_group
    where id_group = {group_id}
    '''
    cur.execute(posts_query)
    posts = cur.fetchall()

    con.close()

    data = {}
    data['group_id'] = group_id
    data['posts'] = [i[0] for i in posts]

    with open(f'processed/{group_id}_posts.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

    # возвращаем результат
    return send_file(
        f'processed/{group_id}_posts.json',
        as_attachment=True,
        attachment_filename=f'{group_id}_posts.json',
        cache_timeout=0
    )

@app.route('/search_kw')
def search_kw():
    return render_template('search_kw.html')

@app.route('/process_kw')
def process_kw():
    if not request.args:
        return redirect(url_for('search_page'))

    keyword = request.args.get('group_link')

    con = sqlite3.connect('vk_groups.db', check_same_thread=False)
    cur = con.cursor()

    similar = set()
    cur.execute(f'''SELECT vk_group FROM keywords
            JOIN group_to_keywords ON keywords.id = group_to_keywords.id_keyword
            JOIN groups ON groups.id = group_to_keywords.id_group
            WHERE keywords.keyword = '{keyword}'
        ''')

    for j in cur.fetchall():
        if len(similar) > 5:
            break
        similar.add(j[0])

    con.close()

    return render_template(
        'result_kw.html', similar=similar
        )

#%%
#if __name__ == '__main__':
#    app.run()