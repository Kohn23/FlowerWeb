# Generated by Django 5.1.7 on 2025-04-14 06:37

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="FlowerDetail",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("flower_type", models.CharField(max_length=100, unique=True)),
                ("sunlight", models.CharField(max_length=50)),
                ("water", models.CharField(max_length=50)),
                ("temperature", models.CharField(max_length=50)),
                ("season", models.CharField(max_length=50)),
                ("soil", models.CharField(max_length=100)),
                ("height", models.CharField(max_length=50)),
                ("spread", models.CharField(max_length=50)),
                ("lifespan", models.CharField(max_length=50)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
