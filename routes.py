import os

import numpy as np
import pandas as pd
import wtforms
import pickle
from flask import Blueprint, render_template, request, redirect, url_for, flash, send_file
from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
from wtforms import StringField, SubmitField, TextAreaField
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

DATA_PATH = 'train.csv'
ATTRS_PATH = 'attr.csv'

CATBOOST_PATH = 'cboost_model'
LGBM_PATH = 'lgbm_model.pkl'
SAVE_TO_C = 'submit_cat.csv.gz'
SAVE_TO_L = 'submit_lgbm.csv.gz'
SAVE_TO_E = 'submit_ensemble.csv.gz'

main = Blueprint("main", __name__)


def make_predicts(model, joined_v, df, save_to):
    predicts = model.predict(joined_v.drop(columns='x1'))
    pred_series = pd.Series(predicts, index=df.index)
    df['x1'] = pred_series
    df = df[['ego_id', 'u', 'v', 'x1']]
    df.to_csv(save_to, index=False, compression='gzip')
    return predicts


class CreateForm(FlaskForm):
    # title = StringField('Title')
    file = wtforms.FileField('File')
    submit = SubmitField('Upload')


@main.route("/", methods=["POST", "GET"])
def train():
    form = CreateForm()

    if request.method == "POST":
        file_buff = None
        for file in request.files.getlist('uploads'):
            file_buff = file
            file.save(file.filename)

        df = pd.read_csv(DATA_PATH)
        attrs = pd.read_csv(ATTRS_PATH)

        joined_u = pd.merge(df, attrs, left_on=['ego_id', 'u'], right_on=['ego_id', 'u'], how='left')
        joined_v = pd.merge(joined_u, attrs, left_on=['ego_id', 'v'], right_on=['ego_id', 'u'], how='left',
                            suffixes=('_u', '_v'))

        joined_v = joined_v[[
            'ego_id', 'u_u', 'v', 't', 'x1', 'x2', 'x3', 'age_u', 'city_id_u',
            'sex_u', 'school_u', 'university_u', 'age_v', 'city_id_v',
            'sex_v', 'school_v', 'university_v'
        ]]
        joined_v.columns = [
            'ego_id', 'u', 'v', 't', 'x1', 'x2', 'x3', 'age_u', 'city_id_u',
            'sex_u', 'school_u', 'university_u', 'age_v', 'city_id_v',
            'sex_v', 'school_v', 'university_v'
        ]

        y_train = joined_v['x1']
        X_train = joined_v.drop(columns=['x1'])

        lgbm = LGBMRegressor(random_state=42, max_depth=10, n_estimators=30)
        cb_reg_1 = CatBoostRegressor(random_seed=13, verbose=200)

        lgbm.fit(X_train, y_train)
        cb_reg_1.fit(X_train, y_train)

        with open('lgbm_model.pkl', 'wb') as file:
            pickle.dump(lgbm, file)

        cb_reg_1.save_model('cboost_model')

    # return send_file(file_buff.filename)

    return render_template('train.html', form=form)


@main.route("/predict", methods=["POST", "GET"])
def predict():
    form = CreateForm()

    if request.method == "POST":
        file_buff = None
        for file in request.files.getlist('uploads'):
            file_buff = file
            file.save(file.filename)

        df = pd.read_csv(DATA_PATH)
        attrs = pd.read_csv(ATTRS_PATH)

        joined_u = pd.merge(df, attrs, left_on=['ego_id', 'u'], right_on=['ego_id', 'u'], how='left')
        joined_v = pd.merge(joined_u, attrs, left_on=['ego_id', 'v'], right_on=['ego_id', 'u'], how='left',
                            suffixes=('_u', '_v'))

        joined_v = joined_v[[
            'ego_id', 'u_u', 'v', 't', 'x1', 'x2', 'x3', 'age_u', 'city_id_u',
            'sex_u', 'school_u', 'university_u', 'age_v', 'city_id_v',
            'sex_v', 'school_v', 'university_v'
        ]]
        joined_v.columns = [
            'ego_id', 'u', 'v', 't', 'x1', 'x2', 'x3', 'age_u', 'city_id_u',
            'sex_u', 'school_u', 'university_u', 'age_v', 'city_id_v',
            'sex_v', 'school_v', 'university_v'
        ]

        model_L = pickle.load(open(LGBM_PATH, 'rb'))

        model_C = CatBoostRegressor()
        model_C.load_model(CATBOOST_PATH)

        predicts_L = make_predicts(model_L, joined_v, df, SAVE_TO_L)
        predicts_С = make_predicts(model_C, joined_v, df, SAVE_TO_C)

        predicts_E = np.add(predicts_L, predicts_С) / 2

        pred_series_ENSEMBLE = pd.Series(predicts_E, index=df.index)

        df['x1'] = pred_series_ENSEMBLE
        df = df[['ego_id', 'u', 'v', 'x1']]

        df.to_csv(SAVE_TO_E, index=False, compression='gzip')

        return send_file(SAVE_TO_E)

    return render_template('predict.html', form=form)
