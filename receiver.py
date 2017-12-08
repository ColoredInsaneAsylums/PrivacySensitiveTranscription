# This program takes the list of classifications crowdsourced in Zooniverse and updates the same in MongoDB corresponding to the subjects.

from panoptes_client import SubjectSet, Subject, Project, Panoptes
from pymongo import MongoClient

import argparse
import csv
import datetime
import io
import itertools
import json

import csh_db_config
import zooniverse_config

def main():
    # connect to zooniverse
    Panoptes.connect(username=zooniverse_config.Zooniverse_USERNAME, password=zooniverse_config.Zooniverse_PASS)
    project = Project.find(zooniverse_config.Project_ID)

    # connection to mongodb
    mongoConn = MongoClient(csh_db_config.DB_HOST + ":" + str(csh_db_config.DB_PORT))
    cshTransDB = mongoConn[csh_db_config.TRANSCRIPTION_DB_NAME]
    cshTransDB.authenticate(csh_db_config.TRANSCRIPTION_DB_USER,
                            csh_db_config.TRANSCRIPTION_DB_PASS)
    cshCollection = cshTransDB[csh_db_config.TRANS_DB_MeetingMinColl]
    cshSubjectSets = cshTransDB[csh_db_config.TRANS_DB_SubjectSets]

    classification_export = Project(zooniverse_config.Project_ID).get_export('classifications')
    classification = classification_export.content.decode('utf-8')

    # keep track of the number of classifications
    num_classifications = 0

    # traverses through each row of classifications and assigns them to appropriate headers
    for row in csv.DictReader(io.StringIO(classification)):
        annotations = json.loads(row['annotations'])
        subject_data = json.loads(row['subject_data'])
        transcription_question_1 = ''
        transcription_text_1 = ''
        transcription_question_2 = ''
        transcription_text_2 = ''
        transcription_filename = ''

        subject_id = row['subject_ids']
        subject_id = str(subject_id)

        # parse the JSON output from Zooniverse into individual fields
        for task in annotations:
            try:
                if 'Is there a word in this image?' in task['task_label']:
                    if task['value'] is not None:
                        transcription_question_1 = str(task['task_label'])
                        transcription_text_1 = str(task['value'])
                        num_classifications += 1
            except KeyError:
                try:
                    if 'Please type the word(s) that appears in this image' in task['task_label']:
                        if task['value'] is not None:
                            transcription_question_2 = str(task['task_label'])
                            transcription_text_2 = str(task['value'])
                except KeyError:
                    continue

            # retrieve and update the record from MongoDB
            updateQuery = {
                '$set':{
                    'responses': [{
                        'labellerId': row['user_id'],
                        'type'      : transcription_text_1,
                        'label'     : transcription_text_2
                    }],
                    'transcription': {
                        'status'   : 'done'
                    }
                }
            }
            record = cshCollection.find_one_and_update({'_id': transcription_filename}, updateQuery)

    print('{} classifications retrieved from Zooniverse and records updated in MongoDB'.format(num_classifications))

if __name__ == '__main__':
    main()

