"""
events.csv:
    event_id,device_id,timestamp,longitude,latitude

app_events.csv:
    event_id,app_id,is_installed,is_active

gender_age_train.csv:
    device_id,gender,age,group

app_labels.csv:
    app_id,label_id

label_categories.csv:
    label_id,category

phone_brand_device_model.csv:
    device_id,phone_brand,device_model
    # note that phone_brand and device_model may include chinese characters

gender_age_test.csv:
    device_id

sample_submission.csv:
    device_id,F23-,F24-26,F27-28,F29-32,F33-42,F43+,M22-,M23-26,M27-28,M29-31,M32-38,M39+
"""