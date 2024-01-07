select
  request_id,
-- from `yc-data-science.one_datamart.all`
  from `yc-data-science.pt.datamart_all_snapshot_pt_rows_demands_2023_08_24`
where
  business_context.business_unit_code = 'PT' and
  business_context.partner_code = 'YOUNITED' and
  -- we are taking a date before 2023-01-01, because the dates between `yuc-pr-dataanalyst.intermediate.transfo_global_business`
  -- decision_timeline.application_date and application_date seem to differ
  application_date > '2023-01-01'
  and application_date < '2023-07-01'

--   (application_status.granting.decision_status = 'granted' or application_date >= timestamp_sub(current_timestamp, interval 90 day)) and
--   application_status.preapproval.preapproval_status is not null
--   1=1
