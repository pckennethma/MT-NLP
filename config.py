api_url = "http://api.conceptnet.io"

human_indicator = {'/c/en/people', '/c/en/person', '/c/en/human', '/c/en/employee', '/c/en/human_adult'}

no_human_indicator = {'/c/en/robot', '/c/en/machine', '/c/en/automation', '/c/en/character', '/c/en/community',
                      '/c/en/place', '/c/en/condition', '/c/en/body_part', '/c/en/organization', '/c/en/event',
                      '/c/en/connection', '/c/en/disease', '/c/en/material', '/c/en/activity', '/c/en/concept',
                      '/c/en/organ', '/c/en/scenery', '/c/en/motion', '/c/en/energy', '/c/en/subject', '/c/en/property',
                      '/c/en/state', '/c/en/plant', '/c/en/class', '/c/en/element', '/c/en/chemical', '/c/en/name',
                      '/c/en/case', '/c/en/object'}

neutral_indicator = {'/c/en/animal', '/c/en/family'}

insensitive_blacklist = {'sort', 'life', 'self', 'family', 'living', 'case', 'nudity', 'link', 'everyone'}