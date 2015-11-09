import json

SETTINGS_FILE="settings.json"
SETTINGS = {}

def init():
  global SETTINGS
  global SETTINGS_FILE
  with open(SETTINGS_FILE) as settings_file:
    read_settings = settings_file.read()
#    print read_settings
    SETTINGS = json.loads(read_settings)

def update(new_settings):
  global SETTINGS
  global SETTINGS_FILE

  for k in new_settings:
    SETTINGS[k]=new_settings[k]

  with open(SETTINGS_FILE, 'w') as settings_file:
    json.dump(SETTINGS, settings_file, sort_keys=True, indent=2)

  with open(SETTINGS_FILE) as settings_file:
    read_settings = settings_file.read()
    print read_settings
    SETTINGS = json.loads(read_settings)

  print "Update Success."
  return 0


if __name__ == "__main__":
  init()
 # print SETTINGS
else:
  init()
