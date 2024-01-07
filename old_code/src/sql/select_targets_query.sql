WITH time_to_responds as (
    SELECT
    contract_reference,
    EXTRACT(HOUR FROM decision_timeline.decision_date - decision_timeline.application_date) as tim_in_hours
    FROM `yuc-pr-dataanalyst.intermediate.transfo_global_business`as dm
    WHERE dm.business_context.business_unit_code='PT'
),

incidents as (with step0 as (
    select dm.contract_reference,
           dm.application_date,
           ARRAY((SELECT AS STRUCT ARRAY((SELECT AS STRUCT inc.* FROM UNNEST(cor.credit_life_events.incidents) AS inc
                  WHERE CAST(inc.reporting_date AS TIMESTAMP) < dm.application_date
                  ORDER BY inc.month)) AS inc
          FROM UNNEST(dm.correlations) AS cor)) AS correlations
          FROM `yc-data-science.one_datamart.all` AS dm
          WHERE dm.business_context.business_unit_code = "FR"
          AND dm.application_status.granting.decision_status = "granted")
    SELECT s0.* EXCEPT (correlations, application_date),
        (SELECT MAX((SELECT IFNULL(MAX(inc.inx), 0) FROM UNNEST(inc) AS inc)) FROM UNNEST(s0.correlations) AS cor) AS corr__total_incidents,
        (SELECT SUM((SELECT IFNULL(MAX(inc.inx), 0) FROM UNNEST(inc) AS inc)) FROM UNNEST(s0.correlations) AS cor) AS corr__largest_incident,
    FROM step0 AS s0),
dedup_past_demands1 AS (
  SELECT
    contract_reference,
    COUNT(1) AS n
  FROM
    `yc-data-science.one_datamart.all`
  GROUP BY
    contract_reference
  HAVING
    n < 2 ),
  dedup_past_demands2 AS (
  SELECT
    request_id,
    COUNT(1) AS n
  FROM
    `yc-data-science.one_datamart.all`
  GROUP BY
    request_id
  HAVING
    n < 2 ),
  step1_past_demands AS (
  SELECT
    ARRAY(
    SELECT
      AS STRUCT *
    FROM
      fpe.correlations AS x
    WHERE
      x.contract_reference != fpe.contract_reference) AS correlations,
    fpe.* EXCEPT(correlations)
  FROM
    `yc-data-science.one_datamart.all` AS fpe
  INNER JOIN
    dedup_past_demands1
  ON
    dedup_past_demands1.contract_reference = fpe.contract_reference
  INNER JOIN
    dedup_past_demands2
  ON
    dedup_past_demands2.request_id = fpe.request_id),
  step2_past_demands AS (
  SELECT
    * EXCEPT (correlations),
    ARRAY_LENGTH(ARRAY(
      SELECT
        AS STRUCT *
      FROM
        step1_past_demands.correlations AS x
      WHERE
        x.application_status.granting.decision_status = "granted"
        AND DATE_DIFF(DATE_TRUNC(step1_past_demands.application_date, DAY), DATE_TRUNC(x.application_date, DAY), DAY) <= 120)) AS number_of_previously_funded_applications,


    ARRAY_LENGTH(ARRAY(
      SELECT
        AS STRUCT *
      FROM
        step1_past_demands.correlations AS x
      WHERE
        DATE_DIFF(DATE_TRUNC(step1_past_demands.application_date, DAY), DATE_TRUNC(x.application_date, DAY), DAY) <= 120)) AS number_of_applications_before_funding,


    (ARRAY_LENGTH(ARRAY(
        SELECT
          AS STRUCT *
        FROM
          step1_past_demands.correlations AS x
        WHERE
          x.application_status.granting.decision_status = "granted")) / (1+(
        SELECT
          MAX(DATE_DIFF(DATE_TRUNC(step1_past_demands.application_date, DAY), DATE_TRUNC(x.application_date, DAY), DAY))
        FROM
          step1_past_demands.correlations AS x
        WHERE
          x.application_status.granting.decision_status = "granted"))) AS freq_previously_funded_applications_per_days
  FROM
    step1_past_demands),


FINAL_CORRELATIONS AS(
    SELECT contract_reference,
    freq_previously_funded_applications_per_days,
    number_of_applications_before_funding,
    number_of_previously_funded_applications,
    ii.corr__total_incidents,
    ii.corr__largest_incident
    FROM step2_past_demands
    LEFT JOIN incidents as ii
    USING(contract_reference)
),

DECISION_STATUS AS(
    select
    contract_reference,
    application_status.granting.decision_status
from `yc-data-science.one_datamart.all`
where
  business_context.business_unit_code = 'PT' and
  business_context.partner_code = 'YOUNITED' and
  (application_status.granting.decision_status = 'granted' or application_date >= timestamp_sub(current_timestamp, interval 90 day)) and
  application_status.preapproval.preapproval_status is not null)


SELECT
  contract_reference,
  corr.*,
  tr.tim_in_hours,
  dn2_4,
  dn2_6,
  dn3_12,
  verified_contact.nationality_code,
  verified_contact.gender_code,
  identification.is_repeat_business,
  loan_info.maturity,
  loan_info.borrowed_capital,
  loan_info.has_coborrower,
  loan_info.is_counter_offer,
  debt_consolidation.is_debt_consolidation,
  acquisition.channel_marketing,
  acquisition.subchannel_marketing,
  DECISION_STATUS.decision_status
FROM
  `yuc-pr-dataanalyst.risk_csm.default_rates` as a
inner join DECISION_STATUS using(contract_reference)
inner join `yuc-pr-dataanalyst.intermediate.transfo_enriched_with_eulerian` b using(contract_reference)
left join time_to_responds as tr
using(contract_reference)
left join FINAL_CORRELATIONS as corr
using(contract_reference)
where
  a.business_unit_code = 'PT'