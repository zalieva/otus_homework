# -*- coding: utf-8 -*-

import os
import googleapiclient.discovery
import csv

DEVELOPER_KEY = "AIzaSyAduJEMTY0oqLAa6q3gmMH1bnni0tM5dD4"
VIDEO_ID = "RIrYWhjdK_o"

# Функция для скачивания даты выхода видео по id
def get_video_date_published(video_id):
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = DEVELOPER_KEY)

    request = youtube.videos().list(
        part="snippet,contentDetails,statistics",
        id=video_id
    )
    response = request.execute()

    return response.get("items")[0].get("snippet").get("publishedAt")

videoPublishedAt = get_video_date_published(VIDEO_ID)
with open(f'datasets/videoData_{VIDEO_ID}.csv', 'w',
          encoding="utf-8") as csv_file:  # конструкция with, чтобы файл закрылся автоматом после всех команд
    writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL,
                        lineterminator='\r')
    # Сохраняем комментарии и дату публикации видео в файл csv

    row = [videoPublishedAt]

    writer.writerow(row)