import dataclasses
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from schemas import StatusEnum, ResponseModel
from core.config import settings


tokenizer = AutoTokenizer.from_pretrained(settings.TOKENIZER)
model = AutoModelForTokenClassification.from_pretrained(settings.MODEL)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)


regexes = [
            {"CREDIT_CARD_RE": r"(?:(4\\d{3}[-\\s]?\\d{4}[-\\s]?\\d{4}[-\\s]?\\d{4})|(3[47]\\d{2}[-\\s]?\\d{6}[-\\s]?\\d{5})|(3(?:0[0-5]|[68]\\d)\\d{11}))"},
            {"UUID": r"[a-f0-9]{8}\-[a-f0-9]{4}\-[a-f0-9]{4}\-[a-f0-9]{4}\-[a-f0-9]{12}"},
            {"EMAIL_ADDRESS_RE":
                r"\b[A-Za-z0-9._%+-]+(\[AT\]|@)[A-Za-z0-9.-]+(\[DOT\]|\.)[A-Za-z]{2,}\b"},
            {"US_SSN_RE": r"\b\d{3}-\d{2}-\d{4}\b"},
            {"BTC_ADDRESS":  r"(?<![a-km-zA-HJ-NP-Z0-9])[13][a-km-zA-HJ-NP-Z0-9]{26,33}(?![a-km-zA-HJ-NP-Z0-9])"},
            {"URL_RE":  r"(?i)((?:https?://|www\d{0,3}[.])?[a-z0-9.\-]+[.](?:(?:international)|(?:construction)|(?:contractors)|(?:enterprises)|(?:photography)|(?:immobilien)|(?:management)|(?:technology)|(?:directory)|(?:education)|(?:equipment)|(?:institute)|(?:marketing)|(?:solutions)|(?:builders)|(?:clothing)|(?:computer)|(?:democrat)|(?:diamonds)|(?:graphics)|(?:holdings)|(?:lighting)|(?:plumbing)|(?:training)|(?:ventures)|(?:academy)|(?:careers)|(?:company)|(?:domains)|(?:florist)|(?:gallery)|(?:guitars)|(?:holiday)|(?:kitchen)|(?:recipes)|(?:shiksha)|(?:singles)|(?:support)|(?:systems)|(?:agency)|(?:berlin)|(?:camera)|(?:center)|(?:coffee)|(?:estate)|(?:kaufen)|(?:luxury)|(?:monash)|(?:museum)|(?:photos)|(?:repair)|(?:social)|(?:tattoo)|(?:travel)|(?:viajes)|(?:voyage)|(?:build)|(?:cheap)|(?:codes)|(?:dance)|(?:email)|(?:glass)|(?:house)|(?:ninja)|(?:photo)|(?:shoes)|(?:solar)|(?:today)|(?:aero)|(?:arpa)|(?:asia)|(?:bike)|(?:buzz)|(?:camp)|(?:club)|(?:coop)|(?:farm)|(?:gift)|(?:guru)|(?:info)|(?:jobs)|(?:kiwi)|(?:land)|(?:limo)|(?:link)|(?:menu)|(?:mobi)|(?:moda)|(?:name)|(?:pics)|(?:pink)|(?:post)|(?:rich)|(?:ruhr)|(?:sexy)|(?:tips)|(?:wang)|(?:wien)|(?:zone)|(?:biz)|(?:cab)|(?:cat)|(?:ceo)|(?:com)|(?:edu)|(?:gov)|(?:int)|(?:mil)|(?:net)|(?:onl)|(?:org)|(?:pro)|(?:red)|(?:tel)|(?:uno)|(?:xxx)|(?:ac)|(?:ad)|(?:ae)|(?:af)|(?:ag)|(?:ai)|(?:al)|(?:am)|(?:an)|(?:ao)|(?:aq)|(?:ar)|(?:as)|(?:at)|(?:au)|(?:aw)|(?:ax)|(?:az)|(?:ba)|(?:bb)|(?:bd)|(?:be)|(?:bf)|(?:bg)|(?:bh)|(?:bi)|(?:bj)|(?:bm)|(?:bn)|(?:bo)|(?:br)|(?:bs)|(?:bt)|(?:bv)|(?:bw)|(?:by)|(?:bz)|(?:ca)|(?:cc)|(?:cd)|(?:cf)|(?:cg)|(?:ch)|(?:ci)|(?:ck)|(?:cl)|(?:cm)|(?:cn)|(?:co)|(?:cr)|(?:cu)|(?:cv)|(?:cw)|(?:cx)|(?:cy)|(?:cz)|(?:de)|(?:dj)|(?:dk)|(?:dm)|(?:do)|(?:dz)|(?:ec)|(?:ee)|(?:eg)|(?:er)|(?:es)|(?:et)|(?:eu)|(?:fi)|(?:fj)|(?:fk)|(?:fm)|(?:fo)|(?:fr)|(?:ga)|(?:gb)|(?:gd)|(?:ge)|(?:gf)|(?:gg)|(?:gh)|(?:gi)|(?:gl)|(?:gm)|(?:gn)|(?:gp)|(?:gq)|(?:gr)|(?:gs)|(?:gt)|(?:gu)|(?:gw)|(?:gy)|(?:hk)|(?:hm)|(?:hn)|(?:hr)|(?:ht)|(?:hu)|(?:id)|(?:ie)|(?:il)|(?:im)|(?:in)|(?:io)|(?:iq)|(?:ir)|(?:is)|(?:it)|(?:je)|(?:jm)|(?:jo)|(?:jp)|(?:ke)|(?:kg)|(?:kh)|(?:ki)|(?:km)|(?:kn)|(?:kp)|(?:kr)|(?:kw)|(?:ky)|(?:kz)|(?:la)|(?:lb)|(?:lc)|(?:li)|(?:lk)|(?:lr)|(?:ls)|(?:lt)|(?:lu)|(?:lv)|(?:ly)|(?:ma)|(?:mc)|(?:md)|(?:me)|(?:mg)|(?:mh)|(?:mk)|(?:ml)|(?:mm)|(?:mn)|(?:mo)|(?:mp)|(?:mq)|(?:mr)|(?:ms)|(?:mt)|(?:mu)|(?:mv)|(?:mw)|(?:mx)|(?:my)|(?:mz)|(?:na)|(?:nc)|(?:ne)|(?:nf)|(?:ng)|(?:ni)|(?:nl)|(?:no)|(?:np)|(?:nr)|(?:nu)|(?:nz)|(?:om)|(?:pa)|(?:pe)|(?:pf)|(?:pg)|(?:ph)|(?:pk)|(?:pl)|(?:pm)|(?:pn)|(?:pr)|(?:ps)|(?:pt)|(?:pw)|(?:py)|(?:qa)|(?:re)|(?:ro)|(?:rs)|(?:ru)|(?:rw)|(?:sa)|(?:sb)|(?:sc)|(?:sd)|(?:se)|(?:sg)|(?:sh)|(?:si)|(?:sj)|(?:sk)|(?:sl)|(?:sm)|(?:sn)|(?:so)|(?:sr)|(?:st)|(?:su)|(?:sv)|(?:sx)|(?:sy)|(?:sz)|(?:tc)|(?:td)|(?:tf)|(?:tg)|(?:th)|(?:tj)|(?:tk)|(?:tl)|(?:tm)|(?:tn)|(?:to)|(?:tp)|(?:tr)|(?:tt)|(?:tv)|(?:tw)|(?:tz)|(?:ua)|(?:ug)|(?:uk)|(?:us)|(?:uy)|(?:uz)|(?:va)|(?:vc)|(?:ve)|(?:vg)|(?:vi)|(?:vn)|(?:vu)|(?:wf)|(?:ws)|(?:ye)|(?:yt)|(?:za)|(?:zm)|(?:zw))(?:/[^\s()<>]+[^\s`!()\[\]{};:\'\".,<>?\xab\xbb\u201c\u201d\u2018\u2019])?)"},
            {"PHONE_NUMBER_ZH":
                r"(13[0-9]|14[5-9]|15[0-3,5-9]|16[6]|17[0-8]|18[0-9]|19[8,9])\d{8}"},
            {"PHONE_NUMBER_WITH_EXT":  r"(?i)((?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*(?:[2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|(?:[2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?(?:[2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?(?:[0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(?:\d+)?))"},
            {"DATE_RE": r"(?i)(?:(?<!\\:)(?<!\\:\\d)[0-3]?\\d(?:st|nd|rd|th)?\\s+(?:of\\s+)?(?:jan\\.?|january|feb\\.?|february|mar\\.?|march|apr\\.?|april|may|jun\\.?|june|jul\\.?|july|aug\\.?|august|sep\\.?|september|oct\\.?|october|nov\\.?|november|dec\\.?|december)|(?:jan\\.?|january|feb\\.?|february|mar\\.?|march|apr\\.?|april|may|jun\\.?|june|jul\\.?|july|aug\\.?|august|sep\\.?|september|oct\\.?|october|nov\\.?|november|dec\\.?|december)\\s+(?<!\\:)(?<!\\:\\d)[0-3]?\\d(?:st|nd|rd|th)?)(?:\\,)?\\s*(?:\\d{4})?|[0-3]?\\d[-\\./][0-3]?\\d[-\\./]\\d{2,4}"},
            {"TIME_RE":
                r"(?i)\\d{1,2}:\\d{2} ?(?:[ap]\\.?m\\.?)?|\\d[ap]\\.?m\\.?"},
            {"HEX_COLOR": r"(#(?:[0-9a-fA-F]{8})|#(?:[0-9a-fA-F]{3}){1,2})\\b"},
            {"PRICE_RE":
                r"[$]\\s?[+-]?[0-9]{1,3}(?:(?:,?[0-9]{3}))*(?:\\.[0-9]{1,2})?"},
            {"PO_BOX_RE": r"(?i)P\\.? ?O\\.? Box \\d+"}
        ]

skip_categories = [
            'AGE', 'AMOUNT', 'CURRENCY', 'CURRENCY CODE', 'DATE', 'SEX', 'TIME',
            'CURRENCYNAME', 'CURRENCYSYMBOL', 'FIRSTNAME', 'MIDDLENAME', 'FULLNAME',
            'SUFFIX', 'MAC', 'JOBTYPE', 'MASKEDNUMBER', 'JOBDESCRIPTOR'
        ]


@dataclasses.dataclass
class PrivacyChecker:
    threshold: float = settings.THRESHOLD

    def scan(self, text: str):
        results = ResponseModel(status=StatusEnum.PASSED,  # Default to PASSED
                                categories=[])
        self.perform_ner_analysis(text, results)
        self.perform_regex_checks(text, results)
        return results

    def perform_ner_analysis(self, text, results):
        ner_results = ner_pipeline(text, aggregation_strategy="first")
        for result in ner_results:
            if float(result['score']) >= self.threshold and result['entity_group'] not in skip_categories:
                results.status = StatusEnum.FAILED
                results.categories.append({
                    'entity_group': result['entity_group'],
                    'score': float(result['score']),
                    'word': result['word'],
                    'start': result['start'],
                    'end': result['end']
                })

    @staticmethod
    def perform_regex_checks(text, results):
        for regex_dict in regexes:
            for key, regex in regex_dict.items():
                match = re.search(regex, text)
                if match:
                    results.status = StatusEnum.FAILED.value
                    results.categories.append({
                        'entity_group': key,
                        'score': 1.0,  # Default score for regex matches
                        'word': match.group(),
                        'start': match.start(),
                        'end': match.end()
                    })
