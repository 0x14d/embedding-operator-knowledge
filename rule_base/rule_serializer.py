from __future__ import annotations
import datetime
import logging
import sys
from collections import defaultdict
from collections.abc import Iterable
from enum import Enum
from typing import Optional
from collections.abc import Callable
import math

import pandas as pd
from pandas import ExcelWriter

from preprocessing import LabelEncoderForColumns
from .rule import Rule
import parse
from distutils import util

from utilities import Path


class RuleSerializer:
    """
    serializes and deserializes (lists of) Rules into/from csv, Excel or string format.
    Supports the different levels of detail of survey versions (CIKM, KCAP, V3)
    """

    EXCEL_DATA_SHEET_NAME = "Survey"
    EXCEL_METADATA_SHEET_NAME = "Metadata"

    class Format(Enum):
        CSV = 1,
        EXCEL = 2,
        STRING = 3,

    class Version(Enum):
        UNKNOWN = 0,
        CIKM = 1,
        UNQUANTIFIED_INFLUENCES = 2,
        KCAP = 2,  # alias
        QUANTIFIED_INFLUENCES = 3,
        VERSION3 = 3,  # alias

    def serialize(self,
                  rules: Iterable[Rule],
                  output_format: RuleSerializer.Format,
                  version: RuleSerializer.Version | None = None,
                  path: Path | None = None,
                  **kwargs: str | bool | list[str]) -> Optional[list[str]]:
        """
        serialize a list of rules into the specified output format.
        @param rules: list of Rule objects to serialize
        @param output_format: output format to serialize to
        @param version: which rule version to use (CIKM, KCAP, V3)
        @param path: Path to output file (only for csv, Excel)
        @param kwargs: supported kwargs for Excel output:
            | "additional_data" string for second metadata excel sheet
            | "header" list of str: column headers (names) in main excel sheet

        @return: when 'output_format == RuleSerializer.Format.STRING', a list of strings is returned, otherwise none
        """
        _version: RuleSerializer.Version = RuleSerializer.Version.UNKNOWN if version is None else version
        serializer = self._get_serializer(output_format)
        return serializer(rules, _version, path, **kwargs)

    def _get_serializer(self, output_format: RuleSerializer.Format):
        """ select serialization method based on output format """
        if output_format == RuleSerializer.Format.CSV:
            return self._serialize_to_csv
        elif output_format == RuleSerializer.Format.EXCEL:
            return self._serialize_to_excel
        elif output_format == RuleSerializer.Format.STRING:
            return self._serialize_to_string
        else:
            raise ValueError(output_format)

    @staticmethod
    def _serialize_to_csv(rules: Iterable[Rule],
                          version: RuleSerializer.Version,
                          path: Path | None,
                          **kwargs):
        """ serialize to V1 CIKM output format"""
        raise NotImplementedError("use EXCEL instead.")

    @staticmethod
    def _serialize_to_excel(rules: Iterable[Rule],
                            version: RuleSerializer.Version,
                            _path: Path | None,
                            **kwargs) -> None:
        """
        serialize list of rules to excel file
        @param rules: list of rules to serialize
        @param version: which rule version to use (CIKM, KCAP, V3)
        @param _path: Path to output file (only for csv, Excel)
        @param kwargs: supported kwargs:
            | "additional_data" string for second metadata excel sheet
            | "header" for column headers (names) in main excel sheet
        """
        date = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d_%H%M%S")
        path = _path if _path is not None else f"Expert_survey_{date}.xlsx"
        if version == RuleSerializer.Version.UNKNOWN:
            logging.info(f"unknown version! using latest version: {RuleSerializer.Version.VERSION3}")
            version = RuleSerializer.Version.VERSION3

        version_str = version.name if isinstance(version, RuleSerializer.Version) else str(version)
        additional_data = kwargs.get('additional_data', "")
        date = datetime.datetime.today()
        metadata = pd.DataFrame(
            [{
                "Version": version_str,
                "Date": date,
                "Additional Data": additional_data
            }]
        )
        header = kwargs.get('header', None)
        df_header = pd.DataFrame(columns=header) if header is not None else pd.DataFrame()

        str_rules = RuleSerializer._serialize_to_string(rules, version=version)
        df = pd.DataFrame(str_rules, columns=['Rule Candidate'])
        complete_df = pd.concat([df_header, df])

        with ExcelWriter(path) as writer:
            complete_df.to_excel(writer, sheet_name=RuleSerializer.EXCEL_DATA_SHEET_NAME)
            metadata.to_excel(writer, sheet_name=RuleSerializer.EXCEL_METADATA_SHEET_NAME)

    @staticmethod
    def _rule_to_str_v3(rule: Rule) -> str:
        string = ""
        if math.isnan(rule.step_size) and math.isnan(rule.range_start) and math.isnan(rule.range_end):
            string = ""
        elif abs(rule.step_size) > 0:
            string = f' by {abs(rule.step_size):.2f}'
        elif rule.action_quantifier is not None:
            if rule.parameter_type == Rule.ParamType.STRING:
                try:
                    string = f' to {rule.label_encoder.inverse_transform(rule.action_quantifier, rule.parameter)}'
                except IndexError as e:
                    logging.exception(f"index error at rule {repr(rule)},  w/ label encoder {rule.label_encoder} and action quantifier {rule.action_quantifier}, parameter {rule.parameter} of type {rule.parameter_type}")

            else:
                string = f' to {rule.action_quantifier}'
        else:
            if math.isclose(rule.range_start, rule.range_end):
                string = f' to {rule.range_start:.2f}'
            else:
                string = f' in range {rule.range_start:.2f} to {rule.range_end:.2f}'
        additional_string = string

        return f"If you encounter {rule.condition} in [{rule.condition_range[0]:.2f}, {rule.condition_range[1]:.2f}], try to {str(rule.action)} the " \
               f"parameter {rule.parameter}" + additional_string
    
    # these are the backcompat functions
    @staticmethod
    def _rule_to_str_cikm(rule: Rule) -> str:
        raise NotImplementedError("deprecated. use KCAP or Version3.")

    @staticmethod
    def _rule_to_str_kcap(rule: Rule) -> str:
        additional_string = ""
        if math.isnan(rule.step_size) and math.isnan(rule.range_start) and math.isnan(rule.range_end):
            additional_string = ""
        elif abs(rule.step_size) > 0:
            additional_string = f' by {round(abs(rule.step_size), 2)}'
            # TODO XAI-607 enable after allowing mutliple conditions to get more meaningful ranges
            # if rule.range_start != rule.range_end:
            # additional_string = f' by {round(abs(rule.step_size), 2)} in range {round(rule.range_start, 2)} to {round(rule.range_end, 2)}'
        elif rule.action_quantifier is not None:
            if rule.parameter_type == Rule.ParamType.STRING:
                additional_string = f' to {rule.label_encoder.inverse_transform(rule.action_quantifier, rule.parameter)}'
            else:
                additional_string = f' to {rule.action_quantifier}'
        elif rule.range_start != rule.range_end:
            additional_string = f' in range {round(rule.range_start, 2)} to {round(rule.range_end, 2)}'
        elif rule.range_start == rule.range_end:
            additional_string = f' to {round(rule.range_start, 2)}'

        return f"If you encounter {rule.condition}, try to {str(rule.action)} the " \
            f"parameter {rule.parameter}" + additional_string
    
    # aliases
    _rule_to_str_quantified_influences = _rule_to_str_v3
    _rule_to_str_unquantified_influences = _rule_to_str_kcap
    
    @classmethod
    def _serialize_to_string(cls,
                             rules: Iterable[Rule],
                             version: RuleSerializer.Version,
                             _path: Path | None = "",
                             **kwargs) -> list[str]:
        if version == RuleSerializer.Version.UNKNOWN:
            logging.info(f"unknown version! using latest version: {RuleSerializer.Version.VERSION3}")
            version = RuleSerializer.Version.VERSION3

        rule_to_str_fun: Callable[[Rule], str]
        if version == RuleSerializer.Version.CIKM:
            rule_to_str_fun = cls._rule_to_str_cikm
        elif version == RuleSerializer.Version.UNQUANTIFIED_INFLUENCES:
            rule_to_str_fun = cls._rule_to_str_unquantified_influences
        elif version == RuleSerializer.Version.QUANTIFIED_INFLUENCES:
            rule_to_str_fun = cls._rule_to_str_quantified_influences
        else:
            raise NotImplementedError()

        str_rules = [rule_to_str_fun(r) for r in rules]

        return str_rules

    # --- Deserialization ---

    @staticmethod
    def _rule_from_str_v3(string: str, label_encoder: LabelEncoderForColumns) -> Rule:
        # this is the current version, so just return from_string()
        return Rule.from_string(string, label_encoder)

    # these are the backcompat functions
    @staticmethod
    def _rule_from_str_cikm(string: str, label_encoder: LabelEncoderForColumns) -> Rule:
        raise NotImplementedError("deprecated. use KCAP or Version3.")

    @staticmethod
    def _rule_from_str_kcap(string: str, label_encoder: LabelEncoderForColumns) -> Rule:
        rule = Rule(label_encoder)
        result = parse.parse("If you encounter {}, try to {} the parameter {} by {}", string)
        categorical_rule = False
        range_rule = False
        if result is None:
            range_rule = True
            result = parse.parse("If you encounter {}, try to {} the parameter {} in range {} to {}", string)
            if result is None:
                categorical_rule = True
                range_rule = False
                parsed_condition, parsed_action, parsed_parameter, parsed_step = parse.parse("If you encounter {}, try to {} the parameter {} to {}", string)

            else:
                parsed_condition, parsed_action, parsed_parameter, parsed_range_start, parsed_range_end = result
                parsed_step = None
        else:
            parsed_condition, parsed_action, parsed_parameter, parsed_step = result
        try:
            rule.condition = parsed_condition
            try:
                rule.action = Rule.Action.from_string(parsed_action)
            except KeyError as e:
                raise KeyError(
                    'parsing failed - not a validly formatted rule: ' + str(e))
            rule.parameter = parsed_parameter
            if range_rule is False:
                if categorical_rule is True:
                    try:
                        # check whether it's a pseudo categorical rule (happens if there is no intervall but all values are the same)
                        param_value = float(parsed_step)
                        rule.parameter_type = Rule.ParamType.NUMERICAL
                        rule.range_start = param_value
                        rule.range_end = param_value
                    except ValueError:
                        try:
                            #check if its a bool or string
                            rule.action_quantifier = float(util.strtobool(parsed_step))
                            rule.parameter_type = Rule.ParamType.BOOLEAN
                        except ValueError:
                            rule.action_quantifier = label_encoder.transform(parsed_step, rule.parameter)
                            rule.parameter_type = Rule.ParamType.STRING
                else:
                    rule.parameter_type = Rule.ParamType.NUMERICAL
                    if rule.action == Rule.Action.DECREASE_INCREMENTALLY:
                        rule.step_size = -1 * float(parsed_step)
                    else:
                        rule.step_size = float(parsed_step)
            else:
                rule.parameter_type = Rule.ParamType.NUMERICAL
                rule.range_start = float(parsed_range_start)
                rule.range_end = float(parsed_range_end)

        except Exception as e:
            raise ValueError(f'not a valid rule: {string} - {e}')
        return rule

    # aliases
    _rule_from_str_quantified_influences = _rule_from_str_v3
    _rule_from_str_unquantified_influences = _rule_from_str_kcap

    # -----

    def deserialize(self,
                    input_format: RuleSerializer.Format,
                    label_encoder: LabelEncoderForColumns,
                    path: Path | None = None,
                    version: RuleSerializer.Version | None = None,
                    **kwargs: str | bool | list[str]) -> dict[str, list[Rule]] | list[Rule]:
        """
        deserialize Rules from a specified input format into a list/dict of rules.
        @param input_format: input format to deserialize from
        @param label_encoder: LabelEncoderForColumns
        @param path: path to input file, only for csv and Excel
        @param version: which rule version to use (CIKM, KCAP, V3)
        @param kwargs: supported, mandatory kwargs:
            | Excel Input: "usecols" list of column names to deserialize
            | String Input: "strings" list of strings to deserialize
            | CSV Input: "filter_categorical" boolean
        @return: list of Rules, if Format is String or CSV, dict[Column Name] = list[Rules] if Excel
        """
        _version: RuleSerializer.Version = RuleSerializer.Version.UNKNOWN if version is None else version
        deserializer = self._get_deserializer(input_format)
        return deserializer(label_encoder, _version, path, **kwargs)

    def _get_deserializer(self, input_format: RuleSerializer.Format) -> Callable[..., dict[str, list[Rule]] | list[Rule]]:
        if input_format == RuleSerializer.Format.CSV:
            return self._deserialize_from_csv
        elif input_format == RuleSerializer.Format.EXCEL:
            return self._deserialize_from_excel
        elif input_format == RuleSerializer.Format.STRING:
            return self._deserialize_from_string
        else:
            raise ValueError(input_format)

    @staticmethod
    def _deserialize_from_csv(label_encoder: LabelEncoderForColumns,
                              version: RuleSerializer.Version,
                              path: Path,
                              **kwargs) -> list[Rule]:
        filter_categorical = kwargs.get("filter_categorical", False)
        data = pd.read_csv(path, delimiter=';', keep_default_na=False, true_values=['TRUE', 'true', 'True'],
                           false_values=['FALSE', 'False', 'false'])
        list_of_all_rules = []
        for index, row in data.iterrows():
            new_rule = Rule(label_encoder)
            new_rule.condition = row['cura_condition']
            if row['action'] == "":
                continue
            new_rule.action = Rule.Action.from_string(row['action'])
            new_rule.parameter = row['parameter_cura']
            if new_rule.parameter == "":
                continue  # if we do not have an accurate translation for our cura parameter we cannot compare to our kg
            if not row['action_quantifier'] == "":
                if row['categorical'] == 'true':
                    try:
                        # check if its a bool or string
                        new_rule.action_quantifier = float(util.strtobool(row['action_quantifier']))
                        new_rule.parameter_type = Rule.ParamType.BOOLEAN
                    except ValueError:
                        new_rule.action_quantifier = label_encoder.inverse_transform(row['action_quantifier'],
                                                                                     new_rule.parameter)
                        new_rule.parameter_type = Rule.ParamType.STRING
                else:
                    new_rule.parameter_type = Rule.ParamType.NUMERICAL
                    new_rule.action_quantifier = row['action_quantifier']
            if not row['range'] == "":
                csv_range = row['range']  # currently it's a string here
                if '[' in row['range']:
                    # extract range
                    csv_range = csv_range.replace('[', '')
                    csv_range = csv_range.replace(']', '')
                    start, end = csv_range.split(';')
                    new_rule.range_start, new_rule.range_end = float(start), float(end)
                elif '<' in csv_range or '>' in csv_range:
                    # if no range but </> is defined just take the max/min float value as other limit
                    if '<' in csv_range:
                        new_rule.range_start = sys.float_info.min
                        new_rule.range_end = float(csv_range.replace('<', ''))
                    else:
                        new_rule.range_end = sys.float_info.max
                        new_rule.range_start = float(csv_range.replace('>', ''))
            if not row['step_size Cura'] == "":
                new_rule.step_size = float(row['step_size Cura'])

            if filter_categorical is True:
                if not row['categorical'] == 'true':
                    list_of_all_rules.append(new_rule)
            else:
                list_of_all_rules.append(new_rule)
        return list_of_all_rules

    @staticmethod
    def _deserialize_from_excel(label_encoder: LabelEncoderForColumns,
                                version: RuleSerializer.Version,
                                path: Path,
                                **kwargs) -> dict[str, list[Rule]]:
        use_cols = kwargs.get('usecols', None)

        try:
            metadata = pd.read_excel(path, sheet_name=RuleSerializer.EXCEL_METADATA_SHEET_NAME)
            # todo: row 0 only
            excel_version = metadata['Version'][0]
            version = RuleSerializer.Version[excel_version]
            additional_data = metadata['Additional Data'][0]
            date = metadata['Date'][0]
            logging.info(f"read excel file {path} from {date}: Additional Data: {additional_data}")
        except Exception as e:
            logging.warning(f"no excel sheet named {RuleSerializer.EXCEL_METADATA_SHEET_NAME} found. "
                            + "using provided or default configuration.")

        df = pd.read_excel(path, sheet_name=RuleSerializer.EXCEL_DATA_SHEET_NAME, usecols=use_cols)
        rules: defaultdict[str, list[Rule]] = defaultdict()
        if isinstance(use_cols, list):
            for col in use_cols:
                column: pd.Series = df[col].dropna()
                rules[col] = RuleSerializer._deserialize_from_string(label_encoder,
                                                                     version,
                                                                     strings=column)
        else:
            raise TypeError("unknown type for parameter usecols, only list of column labels supported")

        return rules

    @classmethod
    def _deserialize_from_string(cls,
                                 label_encoder: LabelEncoderForColumns,
                                 version: RuleSerializer.Version,
                                 path: Optional[Path] = None,
                                 **kwargs) -> list[Rule]:
        strings = kwargs.get('strings', None)
        if strings is None:
            return []
        if version == RuleSerializer.Version.UNKNOWN:
            logging.info(f"unknown version! using latest version: {RuleSerializer.Version.VERSION3}")
            version = RuleSerializer.Version.VERSION3

        rule_from_str_fun: Callable[[str, LabelEncoderForColumns], Rule]
        if version == RuleSerializer.Version.CIKM:
            rule_from_str_fun = cls._rule_from_str_cikm
        elif version == RuleSerializer.Version.UNQUANTIFIED_INFLUENCES:
            rule_from_str_fun = cls._rule_from_str_unquantified_influences
        elif version == RuleSerializer.Version.QUANTIFIED_INFLUENCES:
            rule_from_str_fun = cls._rule_from_str_quantified_influences
        else:
            raise NotImplementedError()

        rules = [rule_from_str_fun(s, label_encoder) for s in strings]

        return rules
