# iRnWs
Information Retrieval &amp; Web Search

[![Build Status](https://travis-ci.org/ChipaKraken/iRnWs.svg?branch=master)](https://travis-ci.org/ChipaKraken/iRnWs)

```bash
sudo apt-get install ruby1.9.1-dev
sudo gem install travis -v 1.8.8 --no-rdoc --no-ri
wget -qO- https://cli-assets.heroku.com/install-ubuntu.sh | sh
heroku auth:login
travis encrypt $(heroku auth:token) --add deploy.api_key
```