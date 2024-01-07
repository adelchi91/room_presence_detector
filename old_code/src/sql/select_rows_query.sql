select
  request_id,
--   contract_reference,
from `yc-data-science.pt.datamart_all_snapshot_pt_2023_08_10`
--`yc-data-science.one_datamart.all`
where
  business_context.business_unit_code = 'PT' and
  business_context.partner_code = 'YOUNITED' and
  (application_status.granting.decision_status = 'granted' or application_date >= timestamp_sub(current_timestamp, interval 90 day)) and
  application_status.preapproval.preapproval_status is not null
