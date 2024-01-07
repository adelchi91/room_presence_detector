WITH score_bands AS(
    SELECT request_id,  score.score_category_replay, score.score_value_replay, score.score_value_declared, decision_timeline.pre_decision_status, score.score_version_replay,
            business_context.business_unit_code, score.score_category_declared,
            EXTRACT(HOUR FROM decision_timeline.decision_date - decision_timeline.application_date) as tim_in_hours
    FROM `yuc-pr-dataanalyst.intermediate.transfo_global_business`
    WHERE business_context.business_unit_code='PT'
    AND score.score_version_replay='credit_score_pt/v2_1'
    AND decision_timeline.application_date > '2023-01-01'
    and decision_timeline.application_date < '2023-07-01'
    -- AND decision_timeline.application_date < '2023-04-02'
    AND business_context.partner_code='YOUNITED'
        ),
      INELIGIBILITY_REASONS AS(
        SELECT request_id,
    application_status.preapproval.preapproval_reason as preapproval_reasons
     FROM `yc-data-science.one_datamart.all`
      ),

    time_to_responds as (
    SELECT
    request_id,
    EXTRACT(HOUR FROM decision_timeline.decision_date - decision_timeline.application_date) as tim_in_hours
    FROM `yuc-pr-dataanalyst.intermediate.transfo_global_business`as dm
    WHERE dm.business_context.business_unit_code='PT'
    AND score.score_version_replay='credit_score_pt/v2_1'
    AND decision_timeline.application_date > '2023-01-01'
    and decision_timeline.application_date < '2023-07-01'
    -- AND decision_timeline.application_date < '2023-04-02'
    AND business_context.partner_code='YOUNITED'
),

incidents as (with step0 as (
    select dm.request_id,
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
    request_id,
    COUNT(1) AS n
  FROM
    `yc-data-science.one_datamart.all`
  GROUP BY
    request_id
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
      x.request_id != fpe.request_id) AS correlations,
    fpe.* EXCEPT(correlations)
  FROM
    `yc-data-science.one_datamart.all` AS fpe
  INNER JOIN
    dedup_past_demands1
  ON
    dedup_past_demands1.request_id = fpe.request_id
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
    SELECT request_id,
    freq_previously_funded_applications_per_days,
    number_of_applications_before_funding,
    number_of_previously_funded_applications,
    ii.corr__total_incidents,
    ii.corr__largest_incident
    FROM step2_past_demands
    LEFT JOIN incidents as ii
    USING(request_id)
),
PREAPPROVAL_SCORE AS (
    SELECT
    request_id,
    contract_reference,
    application_status.preapproval.credit_score[SAFE_OFFSET(0)].score_value as preapproval_score_value,
    application_status.preapproval.preapproval_reason
    FROM `yc-data-science.one_datamart.all`
    )

SELECT
  score_bands.request_id,
  a.has_already_had_a_younited_loan,
--   PREAPPROVAL_SCORE.contract_reference,
--   corr.*,
  score_bands.tim_in_hours,
--   a.dn2_4,
--   a.dn2_6,
--   a.dn3_12,
--   b.verified_contact.nationality_code,
--   b.verified_contact.gender_code,
--   b.identification.is_repeat_business,
--   b.loan_info.maturity,
--   b.loan_info.borrowed_capital,
--   b.loan_info.has_coborrower,
--   b.loan_info.is_counter_offer,
--   b.debt_consolidation.is_debt_consolidation,
--   b.acquisition.channel_marketing,
--   b.acquisition.subchannel_marketing,
  -- # other
  INELIGIBILITY_REASONS.preapproval_reasons,
    score_bands.score_category_declared,
     score_bands.score_category_replay,
     score_bands.score_value_replay,
     score_bands.score_value_declared, score_bands.pre_decision_status,
     PREAPPROVAL_SCORE.preapproval_score_value,
     PREAPPROVAL_SCORE.preapproval_reason
FROM score_bands
LEFT JOIN `yuc-pr-dataanalyst.risk_csm.default_rates` as a
using(request_id)
-- left join `yuc-pr-dataanalyst.intermediate.transfo_enriched_with_eulerian` b using(request_id)
-- left join FINAL_CORRELATIONS as corr
-- using(request_id)
LEFT JOIN INELIGIBILITY_REASONS
USING(request_id)
LEFT JOIN PREAPPROVAL_SCORE
USING(request_id)
