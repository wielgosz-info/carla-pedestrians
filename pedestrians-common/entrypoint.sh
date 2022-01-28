#!/bin/bash

cd /app

echo "Installing ${PACKAGE:-package}@${COMMIT}..."
SETUPTOOLS_SCM_PRETEND_VERSION="0.0.post0.dev38+${COMMIT}.dirty" /venv/bin/python -m pip install --no-cache-dir -e /app

echo "Executing CMD"
exec "$@"