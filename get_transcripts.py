import datetime
import json
import pathlib
import re

from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

data_path = pathlib.Path(__file__).parent / 'data'
talkcorpus_path = data_path / 'talkcorpus'
with_transcript_path = data_path / 'withtranscript'

MIN_TALK_LENGTH = 8 * 60  # 8 minutes in seconds
MAX_TALK_LENGTH = 20 * 60  # 20 minutes in seconds

audience_response = ['laughter', 'applause', 'cheering', 'none']


def init_webpage(url):
    driver = webdriver.Chrome()
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    return driver, soup


def parse_number(nr_as_str):
    nr_as_str = nr_as_str.strip()
    nr_as_str = nr_as_str.replace('(', '')
    nr_as_str = nr_as_str.replace(')', '')
    nr_as_str = nr_as_str.replace(',', '')

    if any(char in nr_as_str for char in ('K', 'M', 'B')):
        nr_as_str = nr_as_str.replace('.', '')
        nr_as_str = nr_as_str.replace('K', '000')
        nr_as_str = nr_as_str.replace('M', '000000')
        nr_as_str = nr_as_str.replace('B', '000000000')
    return int(nr_as_str)


def get_talkcorpus():
    """Parse Talk Corpus (https://www.researchgate.net/publication/338691522_Talk_Corpus_A_Web-based_Corpus_of_TED_Talks_for_English_Language_Teachers_and_Learners)."""
    url = 'http://talkcorpus.com/'
    driver, soup = init_webpage(url)

    talk_titles = soup.find_all('tr', {'class': 'talk-link'})
    metrics = ['WPM', 'Length', 'NAWL', 'NGSL']  # from td blocks without title attribute

    if not talkcorpus_path.is_dir():
        talkcorpus_path.mkdir()

    # loop through each talk title and extract the metrics and url
    for talk_title in talk_titles:
        data_id = talk_title.get('data-id')
        data_title = talk_title.find('td', {'title': data_id}).text
        metadata = {'title': data_title}
        metrics_data = []
        for td in talk_title.contents:
            if td.has_attr('title'):
                if td.get('title') != data_id:  # talk title and id have already been added
                    metadata['FKRE_rating'] = td.get('title')
                    metadata['FKRE_score'] = td.text
            else:
                metrics_data.append(td.text)
        for i in range(len(metrics)):
            metadata[metrics[i]] = metrics_data[i]

        # make sure talk data panel corresponds to the current talk
        talk = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR,
                                                                               f"tr[data-id='{data_id}']")))
        talk.click()

        # click dropdown to make menu visible to driver
        dropdown = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR,
                                                                                   "#talk-details .dropdown-toggle")))
        dropdown.click()

        talk_url = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR,
                                                                                   "#talk-details a[target='_blank']"))
                                                   ).get_attribute("href")
        metadata['URL'] = talk_url

        filename = f'{data_title}.json'

        # make sure name is valid for any OS
        filename = filename.replace(': ', '_', 1)  # first : separates speaker name and title
        forbidden_chars = r'[<>:/\\|?*]'
        filename = re.sub(forbidden_chars, '', filename)
        filename = filename.replace('\"', '\'')
        with open(talkcorpus_path / filename, 'w') as outfile:
            json.dump(metadata, outfile, indent=4)

    driver.quit()


def complete_talkcorpus(files=None):
    """Add transcript and data from TED website to Talk Corpus."""
    corpus = []
    wrong_length = []
    broken_link = []
    multiple_speakers = []
    has_media = []

    if not with_transcript_path.is_dir():
        with_transcript_path.mkdir()

    if files:
        file_list = files
    else:
        file_list = talkcorpus_path.iterdir()
    for file in file_list:
        with open(file, 'r') as talk:
            metadata = json.load(talk)

        # print(metadata['title'])

        time_obj = datetime.datetime.strptime(metadata['Length'], '%H:%M:%S')
        total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
        if MIN_TALK_LENGTH <= total_seconds <= MAX_TALK_LENGTH:
            metadata['seconds'] = total_seconds
        else:
            wrong_length.append(metadata['title'])
            continue

        speaker = metadata['title'].split(':')[0]
        if '+' in speaker:
            multiple_speakers.append(metadata['title'])
            continue

        try:
            # get views and likes; comments are not present for all talks so are not considered
            driver, soup = init_webpage(metadata['URL'])

            if driver.current_url == 'https://www.ted.com/':
                broken_link.append(metadata['title'])
                continue

            error_divs = driver.find_elements(By.CLASS_NAME, "Error")
            if error_divs and 'HTTP Error' in error_divs[
                0].text:  # Error 404 the talk no longer exists at the given link
                broken_link.append(metadata['title'])
                continue

            views_div = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR,
                                                                                        f"div[data-testid='talk-meta']")))
            views = views_div.text.split()[0]
            metadata['view_count'] = parse_number(views)

            like_div = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH,
                                                                                       "//div[contains(text(), 'Like')]")))
            like_text = like_div.text
            match = re.search(r'\(\d+[KM]\)', like_text)  # see if likes are in div text
            if match:
                likes = match.group(0)
            else:
                likes_count_span = WebDriverWait(like_div, 10).until(
                    EC.presence_of_element_located((By.XPATH, "./span")))
                likes = WebDriverWait(driver, 10).until(EC.visibility_of(likes_count_span)).text.strip()
            metadata['like_count'] = parse_number(likes)

            driver.quit()

            # get transcript phrase by phrase
            driver, soup = init_webpage(metadata['URL'] + '/transcript')

            # make sure spans with transcript are loaded
            spans = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.TAG_NAME, 'span')))
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            transcript = []
            plain_text = ''
            transcript_lines = soup.find_all('span', {'class': 'inline cursor-pointer hover:bg-red-300 css-82uonn'})
            skip_talk = False

            for line in transcript_lines:
                phrase = line.text.replace('\n', ' ')
                stripped = phrase.strip().lower()

                if stripped.startswith('(') and stripped.endswith(')'):  # an annotation
                    if any(keyword in stripped for keyword in ('music', 'singing', 'video')):
                        # talks with music or singing are ignored
                        has_media.append(metadata['title'])
                        skip_talk = True
                        break
                    elif transcript:
                        transcript[-1]['annotation'] = stripped[1:-1]  # remove ()
                    # else the annotation doesn't refer to anything that was just said (eg: applause before the speaker starts the speech) and is ignored

                else:
                    if any(char in stripped for char in
                           ('\u266A', '\u266B', '\u266C', '\u266D', '\u266E', '\u266F')):  # has music notes
                        # talks with music or singing are ignored
                        has_media.append(metadata['title'])
                        skip_talk = True
                        break
                    if re.search(r'\(.*\)', stripped):  # an annotation surrounded by phrases
                        match = re.search(r'^(.*?)\((.*?)\)(.*?)$', stripped)
                        if match:
                            first_phrase = match.group(1)
                            annotation = match.group(2)
                            second_phrase = match.group(3)

                            if any(keyword in annotation for keyword in ('music', 'singing', 'video')):
                                # talks with music or singing are ignored
                                has_media.append(metadata['title'])
                                skip_talk = True
                                break
                            else:
                                if transcript:
                                    plain_text += first_phrase + second_phrase
                                    if first_phrase:
                                        first_phrase = re.split(r'(?<=[.!?])\s', first_phrase)

                                        if re.search(r'[.!?]$', transcript[-1]['sentence'].strip()) \
                                                or re.search(r'[.!?]$', transcript[-1]['sentence'].strip()[:-1]):
                                            # last sentence ended; start a new one
                                            transcript[-1]['sentence'] = transcript[-1]['sentence'].strip()
                                            transcript.append({'sentence': first_phrase[0], 'annotation': 'none'})

                                        else:
                                            # append to the last sentence
                                            transcript[-1]['sentence'] = transcript[-1]['sentence'] + first_phrase[0]

                                    transcript[-1]['annotation'] = annotation

                                    if first_phrase and len(first_phrase) > 1:
                                        # save remaining sentences in fist phrase
                                        for i in range(1, len(first_phrase)):
                                            transcript.append({'sentence': first_phrase[i], 'annotation': 'none'})

                                    if second_phrase:
                                        second_phrase = re.split(r'(?<=[.!?])\s', second_phrase)

                                        for i in range(len(second_phrase)):
                                            # save all sentences in second phrase
                                            transcript.append({'sentence': second_phrase[i], 'annotation': 'none'})

                    else:  # no annotation in phrase
                        plain_text += phrase
                        phrase = re.split(r'(?<=[.!?])\s', phrase)
                        phrase = [sentence for sentence in phrase if sentence != '']
                        if transcript:
                            if re.search(r'[.!?]$', transcript[-1]['sentence'].strip()) \
                                    or re.search(r'[.!?]$', transcript[-1]['sentence'].strip()[
                                                            :-1]):  # some sentences have " after punctuation mark
                                # last sentence ended; start a new one
                                transcript.append({'sentence': phrase[0], 'annotation': 'none'})
                            else:
                                # append to the last sentence
                                transcript[-1]['sentence'] = transcript[-1]['sentence'] + phrase[0]
                        else:
                            transcript.append({'sentence': phrase[0], 'annotation': 'none'})

                        if len(phrase) > 1:
                            for i in range(1, len(phrase)):
                                transcript.append({'sentence': phrase[i], 'annotation': 'none'})

            if skip_talk:
                continue

            metadata['transcript'] = transcript

            driver.quit()

        except Exception as e:
            print(f"Exception occurred for {metadata['title']}")
            print(f"Stack trace: {e}")
            continue

        metadata['FKRE_score'] = float(metadata['FKRE_score'])
        metadata['WPM'] = float(metadata['WPM'])
        metadata['NAWL'] = int(metadata['NAWL'])
        metadata['NGSL'] = int(metadata['NGSL'])
        metadata['raw_transcript'] = plain_text

        corpus.append(metadata)

    with open(data_path / 'corpus.json', 'w', encoding='utf-8') as with_transcript:
        json.dump(corpus, with_transcript, indent=4)

    with open(data_path / '_not_completed.log', 'w', encoding='utf-8') as abandoned:
        if wrong_length:
            abandoned.write('The following talks were too short or too long:\n' + '\n'.join(wrong_length))
            abandoned.write('\n\n')
        if broken_link:
            abandoned.write('The following talks were removed from the TED website:\n' + '\n'.join(broken_link))
            abandoned.write('\n\n')
        if multiple_speakers:
            abandoned.write('The following talks had more than 1 speaker:\n' + '\n'.join(multiple_speakers))
            abandoned.write('\n\n')
        if has_media:
            abandoned.write(
                'The following talks contained music, singing or videos showed by the speaker:\n' + '\n'.join(
                    has_media))
            abandoned.write('\n\n')


def cleanup_corpus():
    corpus = open(data_path / 'corpus.json', 'r', encoding='utf-8').read()

    # Make sure each sentence has a space after its punctuation mark
    pattern = r'(?<=[.,;?!])(?=[^\s])'

    for ted_talk in corpus:
        # Discard talk length as string
        if 'Length' in ted_talk:
            del ted_talk['Length']

        ted_talk['likes_per_view'] = ted_talk['like_count'] / ted_talk['view_count']

        transcript = ted_talk['raw_transcript'].lower()
        # Add a space after each punctuation mark that is followed by a lowercase letter
        transcript = re.sub(pattern, ' ', transcript)

        # Update the transcript in the TED talk dictionary
        ted_talk['raw_transcript'] = transcript


        # Count audience responses
        total_responses = {'laughter': 0, 'applause': 0, 'cheering': 0}

        for sentence in ted_talk['transcript']:
            if ' ' in sentence['annotation']:
                annotations = sentence['annotation'].split()
                annotations.sort()
                parsed_annotation = []

                for annotation in annotations:
                    if annotation in audience_response and annotation != 'none':
                        total_responses[annotation] += 1
                        parsed_annotation.append(annotation)
                if not parsed_annotation:
                    sentence['annotation'] = ['none']
                else:
                    sentence['annotation'] = parsed_annotation
                        
            elif sentence['annotation'] in audience_response:
                if sentence['annotation'] != 'none':
                    total_responses[sentence['annotation']] += 1
                sentence['annotation'] = [sentence['annotation']]
            else:
                # remove annotations that don't describe a reaction from the audience
                sentence['annotation'] = ['none']

        ted_talk['total_responses'] = total_responses


    with open(data_path / 'formatted_corpus.json', 'w') as pretty_corpus:
        json.dump(corpus, pretty_corpus, indent=4)


if __name__ == '__main__':
    get_talkcorpus()
    complete_talkcorpus()
    cleanup_corpus()
