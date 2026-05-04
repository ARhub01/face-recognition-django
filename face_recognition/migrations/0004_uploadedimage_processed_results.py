# Generated for Face Recognition Django System v1.1

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("face_recognition", "0003_uploadedimage_emotion_uploadedimage_emotion_conf"),
    ]

    operations = [
        migrations.AddField(
            model_name="uploadedimage",
            name="processed_image",
            field=models.ImageField(blank=True, null=True, upload_to="processed/"),
        ),
        migrations.AddField(
            model_name="uploadedimage",
            name="face_count",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name="uploadedimage",
            name="results_json",
            field=models.JSONField(blank=True, default=list),
        ),
    ]
