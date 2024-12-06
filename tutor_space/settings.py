from pathlib import Path
import os
from datetime import timedelta
import dj_database_url
from dotenv import load_dotenv

load_dotenv()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.getenv('SECRET_KEY', 'django-insecure-9mr(70e3lqk12+injn$py^o4#h7vs+0bees(mvjr_8%5v%b305')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = int(os.getenv('DEBUG', 0))

ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', '').split(',')

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.sites',

    'containers',
    'profiles',
    'assignments',
    'agents',

    'allauth',
    'allauth.account',
    'allauth.socialaccount',
    'rest_framework',
    'rest_framework.authtoken',
    'rest_framework_simplejwt',
    'corsheaders',
    'dj_rest_auth',
    'dj_rest_auth.registration',
]

IP_ROYAL_PROXY_HOST = 'geo.iproyal.com'
IP_ROYAL_PROXY_PORT = '32325'
IP_ROYAL_PROXY_USER = '8Os7s9tYEIdWg3GM'
IP_ROYAL_PROXY_PASS = 'ubTgmI7Y5iJ5YKPC'
IP_ROYAL_PROXY_COUNTRY = 'us'

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'allauth.account.middleware.AccountMiddleware',
]

ROOT_URLCONF = 'tutor_space.urls'

DOCKER_CONFIG = {
    'TIMEOUTS': {
        'CLIENT': 300,
        'BUILD': 600,     
        'START': 300,    
        'RUN': 1800     
    },
    'RETRIES': {
        'MAX_ATTEMPTS': 3,
        'DELAY': 5
    },
    'RESOURCES': {
        'MEMORY': '512m',
        'CPU_COUNT': 1
    }
}

LANGUAGE_DOCKER_IMAGES = {
    'python': {
        'image': 'python:3.9-slim',
        'repository': 'python',
        'tag': '3.9-slim',
        'default_command': 'python',
        'execution_flags': ['-c'],
        'language_for_execution': 'python',
        'ports': {'5000/tcp': None},
        'environment': {
            'development': {
                'PYTHONUNBUFFERED': '1',
                'PYTHONPATH': '/app',
                'FLASK_ENV': 'development',
                'FLASK_DEBUG': '1'
            },
            'production': {
                'PYTHONUNBUFFERED': '1',
                'PYTHONPATH': '/app',
                'FLASK_ENV': 'production',
                'FLASK_DEBUG': '0'
            }
        }
    },
    'javascript': {
        'image': 'node:16-alpine',
        'repository': 'node',
        'tag': '16-alpine',
        'default_command': 'node',
        'execution_flags': ['-e'],
        'language_for_execution': 'node',
        'ports': {'3000/tcp': None},
        'environment': {
            'development': {
                'NODE_ENV': 'development',
                'PORT': '3000',
                'DEBUG': 'true'
            },
            'production': {
                'NODE_ENV': 'production',
                'PORT': '3000'
            }
        }
    }
}

# Default environment mode
DOCKER_DEFAULT_ENV = os.getenv('DOCKER_ENV', 'development')

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'INFO',
        },
        'containers': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
}

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': ['templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'tutor_space.wsgi.application'

# Database
# https://docs.djangoproject.com/en/5.1/ref/settings/#databases

DATABASES = {
    'default': dj_database_url.parse(os.getenv('DATABASE_URL', 'sqlite:///db.sqlite3'))
}

# Password validation
# https://docs.djangoproject.com/en/5.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'dj_rest_auth.jwt_auth.JWTCookieAuthentication',
    ],
}

CORS_ALLOW_ALL_ORIGINS = False
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOWED_ORIGINS = os.getenv('CORS_ALLOWED_ORIGINS', 'http://localhost:5173').split(',')

REST_USE_JWT = True
JWT_AUTH_COOKIE = 'tutor_space_jwt'
JWT_AUTH_REFRESH_COOKIE = 'tutor_space_jwt_refresh'

REST_AUTH = {
    'USE_JWT': True,
    'JWT_AUTH_COOKIE': 'tutor_space_jwt',
    'JWT_AUTH_REFRESH_COOKIE': 'tutor_space_jwt_refresh',
    'JWT_AUTH_RETURN_EXPIRATION': True,
    'JWT_AUTH_HTTPONLY': False,
}

SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=60),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=1),
}

# Internationalization
# https://docs.djangoproject.com/en/5.1/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.1/howto/static-files/

STATICFILES_DIRS = (
    os.path.join(BASE_DIR,'static'),
)

STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'static')
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

SITE_ID = 1

LOGIN_REDIRECT_URL = "/dashboard/"

AUTHENTICATION_BACKENDS = (
    'django.contrib.auth.backends.ModelBackend',
    'allauth.account.auth_backends.AuthenticationBackend',
)

ACCOUNT_AUTHENTICATION_METHOD = 'email'
ACCOUNT_EMAIL_REQUIRED = True
ACCOUNT_USERNAME_REQUIRED = False
ACCOUNT_EMAIL_VERIFICATION = 'none'

# Default primary key field type
# https://docs.djangoproject.com/en/5.1/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'