"""
Generic functions to get subject metadata using ONE
Jan 2023
Inês Laranjeira
"""

# %%
import pandas as pd
import numpy as np
import datetime
import pickle 
import os

from one.api import ONE
#one = ONE(base_url='https://openalyx.internationalbrainlab.org')  # public database
one = ONE(base_url='https://alyx.internationalbrainlab.org')

# %%

""" 
METADATA FUNCTIONS
"""
def subject_metadata(trials, first_session, matrix):

    """
    Age, weight and weight loss at start of training, sex, housing
    """
    # Query data

    # Initialize output df
    mice = trials['subject_nickname'].unique()
    df = pd.DataFrame(columns=['subject_nickname', 'sex', 'age_start', 
                               'weight_loss', 'weight_start', 'food', 'light',
                               'weekend_water'], index=range(len(subjects)))

    for i, mouse in enumerate(mice):

        nickname = list(trials.loc[trials['subject_nickname'] == mouse, 'subject_nickname'])[0]
        if nickname == 'ibl_witten_13_':
            nickname = 'ibl_witten_13'

        # AGE, START SESSION
        subj_info = one.alyx.rest('subjects', 'list', nickname=nickname)
        dob = subj_info[0]['birth_date']
        date_start = list(first_session.loc[first_session['subject_nickname'] == mouse,
                          'session_date'])[0]
        lab = subj_info[0]['lab']

        # REFERENCE WEIGHT
        if type(date_start) == str:
            restriction = one.alyx.rest('water-restriction', 'list', subject=nickname)
            restriction = pd.DataFrame(restriction)
            restriction['date'] = restriction['start_time'].str[0:10]
            restriction['date'] = pd.to_datetime(restriction['date'], format='%Y-%m-%d')
            possible_dates = restriction.loc[restriction['date'] < date_start, 'date']
            if len(possible_dates) > 0:
                restriction_date = np.max(possible_dates)
                reference_weight = list(restriction.loc[restriction['date'] ==
                                               restriction_date, 'reference_weight'])[0]
            else:
                reference_weight = subj_info[0]['reference_weight']

        # WEIGHT
        weights = one.alyx.rest('weighings', 'list', nickname=nickname)
        weights = pd.DataFrame(weights)
        weights['date'] = weights['date_time'].str[0:10]

        # WEEKEND WATER
        water = one.alyx.rest('water-administrations', 'list', nickname=nickname)
        water = pd.DataFrame(water)
        water['date'] = water['date_time'].str[0:10]
        water['date'] = pd.to_datetime(water['date'], format='%Y-%m-%d')

        # Get training time, to look for weekend water regime under that time
        tr_time = matrix.loc[matrix['subject_nickname'] == mouse, 'date']
        if (type(date_start) == str) & (len(tr_time) > 0):
            water_session = water.loc[water['date'] <= list(tr_time)[0]]

        # Save values
        # SEX
        df['sex'][i] = subj_info[0]['sex']
        df['subject_nickname'][i] = nickname

        # AGE
        if type(date_start) == str:
            df['age_start'][i] = (datetime.date.fromisoformat(date_start) -
                                  datetime.date.fromisoformat(dob)).days
        else:
            print('Starting date missing')

        # WEIGHT
        weight_start = weights.loc[weights['date'] == date_start, 'weight']
        if (len(weight_start) > 0) & (reference_weight > 0):
            df['weight_start'][i] = np.mean(weights.loc[weights['date'] == date_start, 'weight'])
            df['weight_loss'][i] = df['weight_start'][i] / reference_weight

        # WEEKEND WATER
        watertype = list(water_session['water_type'])
        if (('Water 2% Citric Acid' in watertype) or ('Citric Acid Water 2%' in watertype) or
            ('Citric Acid Water 3%' in watertype)):
            df['weekend_water'][i] = 1  # Citric acid
        else:
            df['weekend_water'][i] = 0  # Measured water

        # FOOD
        food_map = {'cortexlab': 18, 'hoferlab': 16, 'mrsicflogellab': 16, 'wittenlab': 20,
                    'mainenlab': 20, 'zadorlab': 20, 'churchlandlab': 20, 'danlab': 20,
                    'angelakilab': 19, 'steinmetzlab': 20}

        df['food'][i] = food_map[lab]

        # LIGHT
        # Inverted cycle = 1; non-inverted = 0 'hoferlab':1, 'mrsicflogellab':1,
        light_map = {'cortexlab': 0, 'wittenlab': 1, 'mainenlab': 0, 'zadorlab': 0, 
                     'churchlandlab': 1, 'danlab': 0, 'angelakilab': 1, 'steinmetzlab': 0,
                     'hoferlab': 1, 'mrsicflogellab': 1}

        df['light'][i] = light_map[lab]

        # Mice in SWC were assigned to diferent light cycles
        light_map_extra = {'SWC_001': 1, 'SWC_002': 1, 'SWC_003': 1, 'SWC_004': 1, 'SWC_005': 1,
                           'SWC_006': 1, 'SWC_007': 1, 'SWC_008': 1, 'SWC_009': 1, 'SWC_010': 1,
                           'SWC_011': 1, 'SWC_012': 1, 'SWC_013': 1, 'SWC_014': 1, 'SWC_015': 1,
                           'SWC_016': 1, 'SWC_017': 1, 'SWC_018': 1, 'SWC_019': 1, 'SWC_020': 1,
                           'SWC_021': 1, 'SWC_022': 1, 'SWC_023': 1, 'SWC_024': 1, 'SWC_025': 1,
                           'SWC_026': 1, 'SWC_027': 1, 'SWC_028': 1, 'SWC_029': 1, 'SWC_030': 1,
                           'SWC_031': 1, 'SWC_032': 1, 'SWC_033': 1, 'SWC_034': 1, 'SWC_035': 1,
                           'SWC_036': 1, 'SWC_044': 1, 'SWC_045': 1, 'SWC_046': 1, 'SWC_050': 1,
                           'SWC_051': 1, 'SWC_052': 1, 'SWC_056': 1, 'SWC_057': 1, 'SWC_058': 1,
                           'SWC_037': 0, 'SWC_038': 0, 'SWC_039': 0, 'SWC_040': 0, 'SWC_041': 0,
                           'SWC_042': 0, 'SWC_043': 0, 'SWC_047': 0, 'SWC_048': 0, 'SWC_049': 0,
                           'SWC_053': 0, 'SWC_054': 0, 'SWC_055': 0, 'SWC_059': 0, 'SWC_060': 0,
                           'SWC_061': 0}
        if nickname in light_map_extra:
            df['light'][i] = light_map_extra[nickname]

    return df


def wheel_metrics(trials, session_no):

    mice = trials['subject_uuid'].unique()
    all_wheel = pd.DataFrame(columns=['subject_nickname', 'subject_uuid', 'disp_norm', 'moves_time'], index=range(len(mice)))

    # Loop through mice
    for m, mouse in enumerate(mice):
        mouse_sessions = trials.loc[trials['subject_uuid'] == mouse]
        nickname = list(mouse_sessions.loc[mouse_sessions['subject_uuid'] == mouse, 'subject_nickname'])[0]
        first_session = np.min(mouse_sessions['training_day'])
            
        mouse_wheel = pd.DataFrame(columns=['subject_nickname', 'subject_uuid', 'training_day', 
        'session_uuid', 'disp_norm', 'moves_time'], index=range(len(mouse_sessions['session_uuid'].unique())))

        # Loop through sessions
        for s, sess in enumerate(mouse_sessions['session_uuid'].unique()):

            # Get eid
            training_day = list(mouse_sessions.loc[mouse_sessions['session_uuid'] == sess, 'training_day'])[0]
            date = str(list(mouse_sessions.loc[mouse_sessions['session_uuid'] == sess, 'session_date'])[0])
            session = list(mouse_sessions.loc[mouse_sessions['session_uuid'] == sess, 'session_number'])[0]
            eid = str(nickname + str('/') + date + str('/00') + str(session))
            if nickname == 'ibl_witten_13_':
                eid = str('ibl_witten_13' + str('/') + date + str('/00') + str(session))

            # Get wheel data
            wheel = one.load_object(eid, 'wheel', collection='alf')

            # Calculate session duration
            session_duration = np.max(mouse_sessions.loc[mouse_sessions['session_uuid'] == sess, 'feedback_times'])
            # Calculate total displacement
            total_displacement = float(np.diff(wheel.position[[0, -1]]))  # total displacement of the wheel during session
            # Calculate total distance
            total_distance = float(np.abs(np.diff(wheel.position)).sum())  # total movement of the wheel

            # Compute values per session
            disp_norm = np.abs(total_displacement / total_distance)
            moves_time = total_distance / session_duration
            # Save values per session
            mouse_wheel['disp_norm'][s] = disp_norm
            mouse_wheel['moves_time'][s] = moves_time
            mouse_wheel['subject_uuid'][s] = mouse
            mouse_wheel['subject_nickname'][s] = nickname
            mouse_wheel['session_uuid'][s] = sess
            mouse_wheel['training_day'][s] = training_day

        # Get averages across 5 sessions and save in all_wheel
        all_wheel['subject_uuid'][m] = mouse
        all_wheel['subject_nickname'][m] = nickname
        if first_session < 2:
            all_wheel['disp_norm'][m] = np.nanmean(mouse_wheel.loc[mouse_wheel['training_day'] <= (session_no + 
                                                        first_session - 1), 'disp_norm'])
            all_wheel['moves_time'][m] = np.nanmean(mouse_wheel.loc[mouse_wheel['training_day'] <= (session_no + first_session - 1),'moves_time'])

    return all_wheel


def ambient_metrics_original(subjects):

    training_sessions_all = training_sessions(subjects)
    mice = training_sessions_all['subject_uuid'].unique()

    all_metrics = pd.DataFrame(columns=['subject_nickname', 'subject_uuid',
                                        'temperature_c', 'relative_humidity', 
                                        'air_pressure_mb'], index=range(len(mice)))
    training_times = training_time(subjects)

    # Loop over mice
    for m, mouse in enumerate(mice[0:2]):

        # Get training time, to look for weekend water regime under that time
        tr_time = training_times.loc[training_times['subject_uuid'] == mouse, 'date']

        if len(tr_time) > 0:
            training_sessions = training_sessions_all.loc[(training_sessions_all['session_date'] <=
                                                           list(tr_time)[0]) &
                                                          (training_sessions_all['subject_uuid'] ==
                                                           mouse)]

            mouse_metrics = pd.DataFrame(columns=['subject_nickname', 'subject_uuid',
                                         'session_uuid', 'training_day', 'temperature_c', 
                                         'relative_humidity', 'air_pressure_mb'], index=range(len(training_sessions
                                                         ['session_uuid'].unique())))

            # Loop over sessions
            for s, sess in enumerate(training_sessions['session_uuid'].unique()):

                training_day = list(training_sessions.loc[training_sessions['session_uuid'] ==
                                                          sess, 'training_day'])[0]
                # Get info for eid
                nickname = list(training_sessions.loc[training_sessions['session_uuid'] == sess,
                                                      'subject_nickname'])[0]
                date = str(list(training_sessions.loc[training_sessions['session_uuid'] == sess,
                                                      'session_date'])[0])
                session = list(training_sessions.loc[training_sessions['session_uuid'] == sess,
                                                     'session_number'])[0]
                eid = str(nickname + str('/') + date + str('/00') + str(session))
                if nickname == 'ibl_witten_13_':
                    eid = str('ibl_witten_13' + str('/') + date + str('/00') + str(session))


                # Get metrics data
                try:
                    metrics = one.load_dataset(eid, '_iblrig_taskData.raw.jsonable', collection='raw_behavior_data')
                    # Compute
                    temp = [m.get('as_data', {}).get('Temperature_C', np.nan) for m in metrics]
                    temp = np.nanmean(temp[1:])
                    press = [m.get('as_data', {}).get('AirPressure_mb', np.nan) for m in metrics]
                    press = np.nanmean(press[1:])
                    hum = [m.get('as_data', {}).get('RelativeHumidity', np.nan) for m in metrics]
                    hum = np.nanmean(hum[1:])

                    # Save
                    mouse_metrics['temperature_c'][s] = temp
                    mouse_metrics['air_pressure_mb'][s] = press
                    mouse_metrics['relative_humidity'][s] = hum

                    mouse_metrics['subject_uuid'][s] = mouse
                    mouse_metrics['subject_nickname'][s] = nickname
                    mouse_metrics['session_uuid'][s] = sess
                    mouse_metrics['training_day'][s] = training_day
                except:
                    print(eid)

        # Get averages across all sessions and save in all_metrics
        all_metrics['subject_uuid'][m] = mouse
        all_metrics['subject_nickname'][m] = nickname
        all_metrics['temperature_c'][m] = np.nanmedian(list(mouse_metrics['temperature_c']))
        all_metrics['relative_humidity'][m] = np.nanmedian(list(mouse_metrics['relative_humidity']))
        all_metrics['air_pressure_mb'][m] = np.nanmedian(list(mouse_metrics['air_pressure_mb']))

    return all_metrics


def ambient_metrics(subjects):

    training_sessions_all = training_sessions(subjects)
    mice = training_sessions_all['subject_uuid'].unique()

    all_metrics = pd.DataFrame(columns=['subject_nickname', 'subject_uuid',
                                        'temperature_c', 'relative_humidity', 
                                        'air_pressure_mb'], index=range(len(mice)))
    training_times = training_time(subjects)

    # Loop over mice
    for m, mouse in enumerate(mice):

        # Get training time, to look for weekend water regime under that time
        tr_time = training_times.loc[training_times['subject_uuid'] == mouse, 'date']

        if len(tr_time) > 0:
            training_sessions = training_sessions_all.loc[(training_sessions_all['session_date'] <=
                                                           list(tr_time)[0]) &
                                                          (training_sessions_all['subject_uuid'] ==
                                                           mouse)]

            mouse_metrics = pd.DataFrame(columns=['subject_nickname', 'subject_uuid',
                                         'session_uuid', 'training_day', 'temperature_c', 
                                         'relative_humidity', 'air_pressure_mb'], index=range(len(training_sessions
                                                         ['session_uuid'].unique())))

            # Loop over sessions
            for s, sess in enumerate(training_sessions['session_uuid'].unique()):

                training_day = list(training_sessions.loc[training_sessions['session_uuid'] ==
                                                          sess, 'training_day'])[0]
                # Get info for eid
                nickname = list(training_sessions.loc[training_sessions['session_uuid'] == sess,
                                                      'subject_nickname'])[0]
                if nickname == 'ZM_1092':
                    all_sessions = len(one.search(subject='ZM_1092'))
                    has_ambient = len(one.search(subject='ZM_1092', data='ambientSensorData'))
                    print(f'{has_ambient}/{all_sessions} sessions have ambient data')
                
                date = str(list(training_sessions.loc[training_sessions['session_uuid'] == sess,
                                                      'session_date'])[0])
                session = list(training_sessions.loc[training_sessions['session_uuid'] == sess,
                                                     'session_number'])[0]
                eid = str(nickname + str('/') + date + str('/00') + str(session))
                if nickname == 'ibl_witten_13_':
                    eid = str('ibl_witten_13' + str('/') + date + str('/00') + str(session))

                has_ambient = one.search(subject=nickname, data='ambientSensorData')

                if str(sess) in has_ambient:

                    amb = one.load_dataset(eid, '_iblrig_ambientSensorData.raw.jsonable')

                    # Compute
                    temp = [m.get('Temperature_C', np.nan) for m in amb]
                    temp = np.nanmean(temp)
                    press = [m.get('AirPressure_mb', np.nan) for m in amb]
                    press = np.nanmean(press)
                    hum = [m.get('RelativeHumidity', np.nan) for m in amb]
                    hum = np.nanmean(hum)

                    # Save
                    mouse_metrics['temperature_c'][s] = temp
                    mouse_metrics['air_pressure_mb'][s] = press
                    mouse_metrics['relative_humidity'][s] = hum

                mouse_metrics['subject_uuid'][s] = mouse
                mouse_metrics['subject_nickname'][s] = nickname
                mouse_metrics['session_uuid'][s] = sess
                mouse_metrics['training_day'][s] = training_day

        # Get averages across all sessions and save in all_metrics
        all_metrics['subject_uuid'][m] = mouse
        all_metrics['subject_nickname'][m] = nickname
        all_metrics['temperature_c'][m] = np.nanmedian(list(mouse_metrics['temperature_c']))
        all_metrics['relative_humidity'][m] = np.nanmedian(list(mouse_metrics['relative_humidity']))
        all_metrics['air_pressure_mb'][m] = np.nanmedian(list(mouse_metrics['air_pressure_mb']))

    return all_metrics
# %%
